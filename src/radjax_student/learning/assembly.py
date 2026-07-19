"""The single production authority for JAX lifecycle assembly."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import (
    ArchitectureConfig,
    ArchitectureInitRequest,
    ArchitectureInitResult,
    ArchitectureRegistry,
)
from radjax_student.architecture.protocols import JaxArchitecturePlugin
from radjax_student.contracts import (
    ObjectiveConfig,
    ObjectiveExecutionDescriptor,
    ObjectiveIdentity,
    ResolvedObjectiveSelection,
)
from radjax_student.learning.composition import (
    build_default_learning_callable_registry,
)
from radjax_student.learning.jax_batch import (
    FiniteJsonJaxBatchMaterializer,
)
from radjax_student.learning.models import LearningState
from radjax_student.objectives import ObjectiveRegistry
from radjax_student.optimizers import (
    OptimizerConfig,
    OptimizerInitRequest,
    OptimizerRegistry,
    validate_jax_optimizer_state,
)
from radjax_student.optimizers.protocols import JaxOptimizerBackend
from radjax_student.runtime import (
    CompilationOptions,
    ExecutionRequest,
    RuntimeCallableReference,
    RuntimeConfig,
    bind_runtime_for_learning,
    build_default_runtime_registry,
    initialization_reference_from_root_seed,
)
from radjax_student.runtime.jax_bridge import materialize_initialization_jax_key
from radjax_student.runtime.registry import RuntimeBackendRegistry
from radjax_student.steps.jax_loop import JaxLearningLifecycle, JaxLoopExecutor

ASSEMBLY_SCHEMA_VERSION = "radjax.jax_learning_assembly.v1"
REQUEST_SCHEMA_VERSION = "radjax.jax_learning_assembly_request.v1"
RESULT_SCHEMA_VERSION = "radjax.jax_learning_assembly_result.v1"
LEARNING_ASSEMBLY_ERROR_CODES = (
    "assembly_request_invalid",
    "assembly_architecture_unknown",
    "assembly_architecture_invalid",
    "assembly_architecture_initialization_failed",
    "assembly_architecture_result_invalid",
    "assembly_objective_unknown",
    "assembly_objective_invalid",
    "assembly_objective_surface_unsupported",
    "assembly_objective_descriptor_invalid",
    "assembly_optimizer_unknown",
    "assembly_optimizer_invalid",
    "assembly_optimizer_initialization_failed",
    "assembly_runtime_unknown",
    "assembly_runtime_invalid",
    "assembly_runtime_context_mismatch",
    "assembly_runtime_key_stream_mismatch",
    "assembly_batch_materializer_unknown",
    "assembly_execution_factory_unknown",
    "assembly_learning_state_invalid",
    "assembly_lifecycle_invalid",
    "assembly_loop_executor_invalid",
    "assembly_identity_mismatch",
)


class LearningAssemblyError(Exception):
    """Stable orchestration failure; component errors remain causal."""

    def __init__(
        self, code: str, detail: str, *, cause: Exception | None = None
    ) -> None:
        if code not in LEARNING_ASSEMBLY_ERROR_CODES:
            raise ValueError(f"unknown learning assembly error code: {code}")
        if not isinstance(detail, str) or not detail:
            raise ValueError("learning assembly error detail must be nonempty")
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")
        if cause is not None:
            self.__cause__ = cause


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _factory(
    mode: str, callable_reference: RuntimeCallableReference
) -> Callable[[LearningState], ExecutionRequest]:
    def make(state: LearningState) -> ExecutionRequest:
        return ExecutionRequest(
            request_id=f"assembly.{state.run_id}.{state.global_step}",
            function_id=callable_reference.callable_id,
            mode=mode,
            compilation_options=CompilationOptions(mode=mode, synchronize_results=True),
            callable_reference=callable_reference,
        )

    return make


@dataclass(frozen=True)
class JaxLearningAssemblyRequest:
    architecture_id: str
    architecture_version: int
    architecture_config: ArchitectureConfig
    objective_identity: ObjectiveIdentity
    objective_config: ObjectiveConfig
    optimizer_id: str
    optimizer_version: int
    optimizer_config: OptimizerConfig
    runtime_backend_id: str
    runtime_implementation_version: str
    runtime_config: RuntimeConfig
    root_seed: int
    learning_state: LearningState
    batch_materializer_id: str = "finite_json.v1"
    execution_request_factory_id: str = "generic.v1"
    precision_policy: str = "float32"
    schedule_values: Mapping[str, Any] = field(default_factory=dict)
    rng_slot: str = "dropout"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.architecture_version, int)
            or isinstance(self.architecture_version, bool)
            or self.architecture_version < 1
        ):
            raise LearningAssemblyError(
                "assembly_architecture_invalid", "architecture_version must be positive"
            )
        if (
            not isinstance(self.architecture_config, ArchitectureConfig)
            or self.architecture_config.architecture_id != self.architecture_id
        ):
            raise LearningAssemblyError(
                "assembly_architecture_invalid",
                "architecture identity and config must match",
            )
        if (
            not isinstance(self.objective_identity, ObjectiveIdentity)
            or not isinstance(self.objective_config, ObjectiveConfig)
            or self.objective_config.identity != self.objective_identity
        ):
            raise LearningAssemblyError(
                "assembly_objective_invalid", "objective identity and config must match"
            )
        if (
            not isinstance(self.optimizer_config, OptimizerConfig)
            or self.optimizer_config.optimizer_id != self.optimizer_id
        ):
            raise LearningAssemblyError(
                "assembly_optimizer_invalid", "optimizer identity and config must match"
            )
        if (
            not isinstance(self.optimizer_version, int)
            or isinstance(self.optimizer_version, bool)
            or self.optimizer_version < 1
        ):
            raise LearningAssemblyError(
                "assembly_optimizer_invalid", "optimizer_version must be positive"
            )
        if (
            not isinstance(self.runtime_config, RuntimeConfig)
            or self.runtime_config.backend_id != self.runtime_backend_id
            or self.runtime_config.seed != self.root_seed
        ):
            raise LearningAssemblyError(
                "assembly_runtime_invalid",
                "runtime identity, config, and root seed must match",
            )
        if not isinstance(self.learning_state, LearningState):
            raise LearningAssemblyError(
                "assembly_learning_state_invalid",
                "learning_state must be LearningState",
            )
        if (
            not isinstance(self.runtime_implementation_version, str)
            or not self.runtime_implementation_version
        ):
            raise LearningAssemblyError(
                "assembly_runtime_invalid", "runtime implementation version is required"
            )
        if any(
            not isinstance(item, str) or not item
            for item in (
                self.architecture_id,
                self.optimizer_id,
                self.runtime_backend_id,
                self.batch_materializer_id,
                self.execution_request_factory_id,
                self.precision_policy,
                self.rng_slot,
            )
        ):
            raise LearningAssemblyError(
                "assembly_request_invalid",
                "assembly identifiers must be nonempty strings",
            )
        if (
            not isinstance(self.root_seed, int)
            or isinstance(self.root_seed, bool)
            or self.root_seed < 0
        ):
            raise LearningAssemblyError(
                "assembly_request_invalid", "root_seed must be a nonnegative integer"
            )
        object.__setattr__(
            self, "schedule_values", MappingProxyType(dict(self.schedule_values))
        )

    def validate(self) -> None:
        """Re-check frozen request invariants at the public assembly boundary."""
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "architecture_id": self.architecture_id,
            "architecture_version": self.architecture_version,
            "architecture_config": self.architecture_config.to_dict(),
            "objective_identity": self.objective_identity.to_dict(),
            "objective_config": self.objective_config.to_dict(),
            "optimizer_id": self.optimizer_id,
            "optimizer_version": self.optimizer_version,
            "optimizer_config": self.optimizer_config.to_dict(),
            "runtime_backend_id": self.runtime_backend_id,
            "runtime_implementation_version": self.runtime_implementation_version,
            "runtime_config": self.runtime_config.to_dict(),
            "root_seed": self.root_seed,
            "learning_state": self.learning_state.to_dict(),
            "batch_materializer_id": self.batch_materializer_id,
            "execution_request_factory_id": self.execution_request_factory_id,
            "precision_policy": self.precision_policy,
            "schedule_values": dict(self.schedule_values),
            "rng_slot": self.rng_slot,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JaxLearningAssemblyRequest:
        required = set(cls.__dataclass_fields__) | {"schema_version"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise LearningAssemblyError(
                "assembly_request_invalid", "request fields are missing or unknown"
            )
        if payload["schema_version"] != REQUEST_SCHEMA_VERSION:
            raise LearningAssemblyError(
                "assembly_request_invalid", "request schema is invalid"
            )
        return cls(
            architecture_id=payload["architecture_id"],
            architecture_version=payload["architecture_version"],
            architecture_config=ArchitectureConfig.from_dict(
                payload["architecture_config"]
            ),
            objective_identity=ObjectiveIdentity.from_dict(
                payload["objective_identity"]
            ),
            objective_config=ObjectiveConfig.from_dict(payload["objective_config"]),
            optimizer_id=payload["optimizer_id"],
            optimizer_version=payload["optimizer_version"],
            optimizer_config=OptimizerConfig.from_dict(payload["optimizer_config"]),
            runtime_backend_id=payload["runtime_backend_id"],
            runtime_implementation_version=payload["runtime_implementation_version"],
            runtime_config=RuntimeConfig.from_dict(payload["runtime_config"]),
            root_seed=payload["root_seed"],
            learning_state=LearningState.from_dict(payload["learning_state"]),
            batch_materializer_id=payload["batch_materializer_id"],
            execution_request_factory_id=payload["execution_request_factory_id"],
            precision_policy=payload["precision_policy"],
            schedule_values=payload["schedule_values"],
            rng_slot=payload["rng_slot"],
        )


@dataclass(frozen=True)
class JaxLearningAssemblyRegistries:
    architecture_registry: ArchitectureRegistry
    objective_registry: ObjectiveRegistry
    optimizer_registry: OptimizerRegistry
    runtime_registry: RuntimeBackendRegistry

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_registry, ArchitectureRegistry):
            raise LearningAssemblyError(
                "assembly_request_invalid",
                "architecture_registry must be ArchitectureRegistry",
            )
        if not isinstance(self.objective_registry, ObjectiveRegistry):
            raise LearningAssemblyError(
                "assembly_request_invalid",
                "objective_registry must be ObjectiveRegistry",
            )
        if not isinstance(self.optimizer_registry, OptimizerRegistry):
            raise LearningAssemblyError(
                "assembly_request_invalid",
                "optimizer_registry must be OptimizerRegistry",
            )
        if not isinstance(self.runtime_registry, RuntimeBackendRegistry):
            raise LearningAssemblyError(
                "assembly_request_invalid",
                "runtime_registry must be RuntimeBackendRegistry",
            )

    @classmethod
    def defaults(
        cls,
        *,
        architecture_registry: ArchitectureRegistry,
        objective_registry: ObjectiveRegistry,
        optimizer_registry: OptimizerRegistry,
    ) -> JaxLearningAssemblyRegistries:
        return cls(
            architecture_registry,
            objective_registry,
            optimizer_registry,
            build_default_runtime_registry(),
        )


@dataclass(frozen=True)
class JaxLearningAssemblyResult:
    lifecycle: JaxLearningLifecycle
    loop_executor: JaxLoopExecutor
    architecture_selection: Mapping[str, str]
    objective_selection: Mapping[str, str]
    optimizer_selection: Mapping[str, str]
    runtime_selection: Mapping[str, str]
    summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, JaxLearningLifecycle):
            raise LearningAssemblyError(
                "assembly_lifecycle_invalid", "result requires a complete lifecycle"
            )
        if not isinstance(self.loop_executor, JaxLoopExecutor):
            raise LearningAssemblyError(
                "assembly_loop_executor_invalid", "result requires a loop executor"
            )
        if self.loop_executor.lifecycle is not self.lifecycle:
            raise LearningAssemblyError(
                "assembly_identity_mismatch", "executor must bind the result lifecycle"
            )
        if not isinstance(
            self.loop_executor.batch_materializer, FiniteJsonJaxBatchMaterializer
        ):
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "executor materializer must be the selected production materializer",
            )
        for value in (
            self.architecture_selection,
            self.objective_selection,
            self.optimizer_selection,
            self.runtime_selection,
            self.summary,
        ):
            if not isinstance(value, Mapping):
                raise LearningAssemblyError(
                    "assembly_request_invalid", "result evidence must be mappings"
                )
        architecture_selection = dict(self.architecture_selection)
        objective_selection = dict(self.objective_selection)
        optimizer_selection = dict(self.optimizer_selection)
        runtime_selection = dict(self.runtime_selection)
        summary = dict(self.summary)
        if architecture_selection != {
            "architecture_id": self.lifecycle.architecture.architecture_id,
            "architecture_version": str(
                self.lifecycle.architecture.architecture_version
            ),
        }:
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "result architecture selection does not match lifecycle",
            )
        if objective_selection != self.lifecycle.objective_selection.to_dict():
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "result objective selection does not match lifecycle",
            )
        if optimizer_selection != {
            "optimizer_id": self.lifecycle.optimizer.optimizer_id,
            "optimizer_version": str(self.lifecycle.optimizer.optimizer_version),
        }:
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "result optimizer selection does not match lifecycle",
            )
        if (
            runtime_selection.get("backend_id")
            != self.lifecycle.runtime_context.backend_id
            or runtime_selection.get("runtime_id")
            != self.lifecycle.runtime_context.runtime_id
            or not isinstance(runtime_selection.get("implementation_version"), str)
        ):
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "result runtime selection does not match lifecycle",
            )
        required_summary = {
            "schema_version",
            "architecture_identity",
            "architecture_version",
            "architecture_config_digest",
            "parameter_catalog_digest",
            "parameter_layout_digest",
            "hf_descriptor_digest",
            "objective_identity",
            "objective_config_digest",
            "objective_surface_digest",
            "objective_implementation_identity",
            "optimizer_identity",
            "optimizer_version",
            "optimizer_config_digest",
            "runtime_backend_identity",
            "runtime_implementation_version",
            "runtime_context_identity",
            "batch_materializer_identity",
            "execution_factory_identity",
            "precision_policy",
            "schedule_digest",
            "rng_slot",
            "runtime_callable_id",
            "runtime_callable_version",
            "runtime_callable_identity_digest",
        }
        if set(summary) != required_summary:
            raise LearningAssemblyError(
                "assembly_identity_mismatch", "result summary fields are invalid"
            )
        expected_summary = {
            "schema_version": ASSEMBLY_SCHEMA_VERSION,
            "architecture_identity": self.lifecycle.architecture.architecture_id,
            "architecture_version": self.lifecycle.architecture.architecture_version,
            "architecture_config_digest": self.lifecycle.config_digest,
            "parameter_catalog_digest": self.lifecycle.catalog_digest,
            "parameter_layout_digest": self.lifecycle.parameter_layout.digest(),
            "hf_descriptor_digest": self.lifecycle.hf_descriptor.digest,
            "objective_identity": self.lifecycle.objective_selection.identity.to_dict(),
            "objective_config_digest": self.lifecycle.objective_config.digest,
            "objective_surface_digest": (
                self.lifecycle.resolved_objective_selection.digest
            ),
            "objective_implementation_identity": (
                self.lifecycle.objective_selection.implementation_identity
            ),
            "optimizer_identity": self.lifecycle.optimizer.optimizer_id,
            "optimizer_version": self.lifecycle.optimizer.optimizer_version,
            "optimizer_config_digest": _digest(
                self.lifecycle.optimizer_config.to_dict()
            ),
            "runtime_backend_identity": self.lifecycle.runtime_context.backend_id,
            "runtime_implementation_version": runtime_selection[
                "implementation_version"
            ],
            "runtime_context_identity": self.lifecycle.runtime_context.runtime_id,
        }
        if any(summary[name] != value for name, value in expected_summary.items()):
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "result summary does not derive from lifecycle identities",
            )
        if not all(
            isinstance(summary[name], str) and len(summary[name]) == 64
            for name in ("schedule_digest", "optimizer_config_digest")
        ):
            raise LearningAssemblyError(
                "assembly_identity_mismatch", "result summary digest is invalid"
            )
        if not all(
            isinstance(summary[name], str) and summary[name]
            for name in (
                "batch_materializer_identity",
                "execution_factory_identity",
                "precision_policy",
                "rng_slot",
            )
        ):
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "result orchestration identities are invalid",
            )
        if (
            self.loop_executor.precision_policy != summary["precision_policy"]
            or self.loop_executor.rng_slot != summary["rng_slot"]
            or summary["batch_materializer_identity"] != "finite_json.v1"
            or summary["execution_factory_identity"] != "generic.v1"
        ):
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "executor orchestration identities do not match the result summary",
            )
        binding = self.loop_executor.runtime_callable_binding
        if binding is None or (
            summary["runtime_callable_id"] != binding.reference.callable_id
            or summary["runtime_callable_version"] != binding.reference.callable_version
            or summary["runtime_callable_identity_digest"]
            != binding.reference.callable_identity_digest
        ):
            raise LearningAssemblyError(
                "assembly_identity_mismatch",
                "runtime callable identity does not match loop binding",
            )
        object.__setattr__(
            self,
            "architecture_selection",
            MappingProxyType(architecture_selection),
        )
        object.__setattr__(
            self,
            "objective_selection",
            MappingProxyType(objective_selection),
        )
        object.__setattr__(
            self,
            "optimizer_selection",
            MappingProxyType(optimizer_selection),
        )
        object.__setattr__(
            self, "runtime_selection", MappingProxyType(runtime_selection)
        )
        object.__setattr__(self, "summary", MappingProxyType(summary))

    @property
    def assembly_digest(self) -> str:
        return _digest(dict(self.summary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "architecture_selection": dict(self.architecture_selection),
            "objective_selection": dict(self.objective_selection),
            "optimizer_selection": dict(self.optimizer_selection),
            "runtime_selection": dict(self.runtime_selection),
            "summary": dict(self.summary),
            "assembly_digest": self.assembly_digest,
        }


def assemble_jax_learning_lifecycle(
    request: JaxLearningAssemblyRequest, *, registries: JaxLearningAssemblyRegistries
) -> JaxLearningAssemblyResult:
    """Assemble one executable lifecycle using only production owner boundaries."""
    if not isinstance(request, JaxLearningAssemblyRequest):
        raise LearningAssemblyError(
            "assembly_request_invalid", "request must be JaxLearningAssemblyRequest"
        )
    request.validate()
    if not isinstance(registries, JaxLearningAssemblyRegistries):
        raise LearningAssemblyError(
            "assembly_request_invalid",
            "registries must be JaxLearningAssemblyRegistries",
        )
    try:
        architecture = registries.architecture_registry.get(request.architecture_id)
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_architecture_unknown", "architecture is not registered", cause=exc
        ) from exc
    if not isinstance(architecture, JaxArchitecturePlugin):
        raise LearningAssemblyError(
            "assembly_architecture_invalid", "architecture lacks JAX capability"
        )
    if architecture.architecture_version != request.architecture_version:
        raise LearningAssemblyError(
            "assembly_architecture_invalid",
            "architecture version does not match request",
        )
    try:
        architecture.validate_config(request.architecture_config)
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_architecture_invalid",
            "architecture configuration is invalid",
            cause=exc,
        ) from exc
    try:
        initialization_reference = initialization_reference_from_root_seed(
            request.root_seed
        )
        initialized = architecture.initialize_parameters(
            ArchitectureInitRequest(
                request.architecture_config,
                initialization_reference.identity,
                request.precision_policy,
                runtime_initialization_material=materialize_initialization_jax_key(
                    initialization_reference.identity
                ),
            )
        )
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_architecture_initialization_failed",
            "architecture initialization failed",
            cause=exc,
        ) from exc
    if not isinstance(initialized, ArchitectureInitResult):
        raise LearningAssemblyError(
            "assembly_architecture_result_invalid",
            "architecture did not return ArchitectureInitResult",
        )
    required_init = (
        initialized.parameter_catalog,
        initialized.parameters,
        initialized.parameter_layout,
        initialized.hf_descriptor,
        initialized.hf_reference,
        initialized.architecture_carry_descriptor,
    )
    if (
        any(item is None for item in required_init)
        or initialized.hf_reference
        != initialized.hf_descriptor.preservation_reference()
    ):
        raise LearningAssemblyError(
            "assembly_architecture_result_invalid",
            "architecture initialization result is incomplete or inconsistent",
        )
    try:
        objective_selection = registries.objective_registry.select(
            request.objective_identity
        )
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_objective_unknown", "objective is not registered", cause=exc
        ) from exc
    try:
        resolved_surface = architecture.resolve_objective_scope(
            request.learning_state.active_objective_scope,
            architecture.architecture_metadata(),
        )
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_objective_surface_unsupported",
            "architecture objective surface resolution failed",
            cause=exc,
        ) from exc
    if not isinstance(resolved_surface, ResolvedObjectiveSelection):
        raise LearningAssemblyError(
            "assembly_objective_surface_unsupported",
            "architecture did not return a resolved surface",
        )
    try:
        objective_selection.plugin.validate_config(request.objective_config)
        objective_selection.plugin.validate_resolved_surface(resolved_surface)
        objective_descriptor = registries.objective_registry.execution_descriptor(
            selection=objective_selection,
            config=request.objective_config,
            resolved_selection=resolved_surface,
        )
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_objective_invalid",
            "objective selection or surface resolution failed",
            cause=exc,
        ) from exc
    if not isinstance(objective_descriptor, ObjectiveExecutionDescriptor):
        raise LearningAssemblyError(
            "assembly_objective_descriptor_invalid",
            "objective registry did not return a descriptor",
        )
    try:
        optimizer = registries.optimizer_registry.get(request.optimizer_id)
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_optimizer_unknown", "optimizer is not registered", cause=exc
        ) from exc
    if not isinstance(optimizer, JaxOptimizerBackend):
        raise LearningAssemblyError(
            "assembly_optimizer_invalid", "optimizer lacks JAX capability"
        )
    if optimizer.optimizer_version != request.optimizer_version:
        raise LearningAssemblyError(
            "assembly_optimizer_invalid", "optimizer version does not match request"
        )
    try:
        optimizer.validate_config(request.optimizer_config)
        envelope = optimizer.initialize_state(
            OptimizerInitRequest(
                request.optimizer_config,
                initialized.parameter_catalog,
                architecture.resolve_update_scope(
                    request.learning_state.active_update_scope,
                    initialized.parameter_catalog,
                ),
            )
        ).optimizer_state
        optimizer_state = optimizer.initialize_jax_state(
            config=request.optimizer_config,
            parameter_layout=initialized.parameter_layout,
            optimizer_state=envelope,
        )
        validate_jax_optimizer_state(
            optimizer_state,
            optimizer=optimizer,
            optimizer_id=request.optimizer_id,
            parameter_layout=initialized.parameter_layout,
            descriptor=optimizer.jax_state_descriptor(initialized.parameter_layout),
        )
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_optimizer_initialization_failed",
            "optimizer state initialization failed",
            cause=exc,
        ) from exc
    try:
        runtime_binding = bind_runtime_for_learning(
            request.runtime_config,
            implementation_version=request.runtime_implementation_version,
            root_seed=request.root_seed,
            rng_slot=request.rng_slot,
            registry=registries.runtime_registry,
        )
    except LookupError as exc:
        raise LearningAssemblyError(
            "assembly_runtime_unknown", "runtime backend is not registered", cause=exc
        ) from exc
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_runtime_invalid", "runtime construction failed", cause=exc
        ) from exc
    backend = runtime_binding.backend
    context = runtime_binding.context
    key_stream = runtime_binding.key_stream
    if context.backend_id != request.runtime_backend_id:
        raise LearningAssemblyError(
            "assembly_runtime_context_mismatch",
            "runtime context belongs to another backend",
        )
    if context.root_seed != request.root_seed:
        raise LearningAssemblyError(
            "assembly_runtime_context_mismatch",
            "runtime context root seed does not match request",
        )
    if key_stream.root_seed != context.root_seed:
        raise LearningAssemblyError(
            "assembly_runtime_key_stream_mismatch",
            "runtime key stream belongs to another seed",
        )
    # Runtime ownership is carried by the lifecycle.  A caller-provided
    # LearningState may intentionally retain a neutral runtime reference.
    state = request.learning_state
    try:
        lifecycle = JaxLearningLifecycle(
            architecture=architecture,
            architecture_config=request.architecture_config,
            architecture_state=initialized.architecture_state,
            architecture_carry=initialized.architecture_carry,
            parameter_catalog=initialized.parameter_catalog,
            parameter_layout=initialized.parameter_layout,
            hf_descriptor=initialized.hf_descriptor,
            hf_reference=initialized.hf_reference,
            objective_selection=objective_selection,
            objective_config=request.objective_config,
            resolved_objective_selection=resolved_surface,
            objective_descriptor=objective_descriptor,
            optimizer=optimizer,
            optimizer_config=request.optimizer_config,
            optimizer_state=optimizer_state,
            parameters=initialized.parameters,
            learning_state=state,
            runtime_context=context,
            runtime_backend=backend,
            runtime_key_stream=key_stream,
            architecture_carry_descriptor=initialized.architecture_carry_descriptor,
        )
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_lifecycle_invalid", "lifecycle construction failed", cause=exc
        ) from exc
    if request.batch_materializer_id != "finite_json.v1":
        raise LearningAssemblyError(
            "assembly_batch_materializer_unknown", "unknown batch materializer"
        )
    if request.execution_request_factory_id != "generic.v1":
        raise LearningAssemblyError(
            "assembly_execution_factory_unknown", "unknown execution request factory"
        )
    try:
        callable_binding = build_default_learning_callable_registry().lookup(
            "radjax.learning.generic_jax_step", 1
        )
        executor = JaxLoopExecutor(
            lifecycle,
            FiniteJsonJaxBatchMaterializer(),
            _factory(
                request.runtime_config.compilation_policy,
                callable_binding.reference,
            ),
            request.precision_policy,
            request.schedule_values,
            request.rng_slot,
            callable_binding,
        )
    except Exception as exc:
        raise LearningAssemblyError(
            "assembly_loop_executor_invalid",
            "loop executor construction failed",
            cause=exc,
        ) from exc
    summary = {
        "schema_version": ASSEMBLY_SCHEMA_VERSION,
        "architecture_identity": request.architecture_id,
        "architecture_version": request.architecture_version,
        "architecture_config_digest": lifecycle.config_digest,
        "parameter_catalog_digest": lifecycle.catalog_digest,
        "parameter_layout_digest": lifecycle.parameter_layout.digest(),
        "hf_descriptor_digest": lifecycle.hf_descriptor.digest,
        "objective_identity": objective_selection.identity.to_dict(),
        "objective_config_digest": request.objective_config.digest,
        "objective_surface_digest": resolved_surface.digest,
        "objective_implementation_identity": (
            objective_selection.implementation_identity
        ),
        "optimizer_identity": optimizer.optimizer_id,
        "optimizer_version": request.optimizer_version,
        "optimizer_config_digest": _digest(request.optimizer_config.to_dict()),
        "runtime_backend_identity": context.backend_id,
        "runtime_implementation_version": request.runtime_implementation_version,
        "runtime_context_identity": context.runtime_id,
        "batch_materializer_identity": request.batch_materializer_id,
        "execution_factory_identity": request.execution_request_factory_id,
        "precision_policy": request.precision_policy,
        "schedule_digest": _digest(dict(request.schedule_values)),
        "rng_slot": request.rng_slot,
        "runtime_callable_id": callable_binding.reference.callable_id,
        "runtime_callable_version": callable_binding.reference.callable_version,
        "runtime_callable_identity_digest": (
            callable_binding.reference.callable_identity_digest
        ),
    }
    return JaxLearningAssemblyResult(
        lifecycle,
        executor,
        MappingProxyType(
            {
                "architecture_id": architecture.architecture_id,
                "architecture_version": str(architecture.architecture_version),
            }
        ),
        MappingProxyType(objective_selection.to_dict()),
        MappingProxyType(
            {
                "optimizer_id": optimizer.optimizer_id,
                "optimizer_version": str(optimizer.optimizer_version),
            }
        ),
        MappingProxyType(
            {
                "backend_id": context.backend_id,
                "runtime_id": context.runtime_id,
                "implementation_version": request.runtime_implementation_version,
            }
        ),
        MappingProxyType(summary),
    )


__all__ = [
    "ASSEMBLY_SCHEMA_VERSION",
    "LEARNING_ASSEMBLY_ERROR_CODES",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "JaxLearningAssemblyRequest",
    "JaxLearningAssemblyRegistries",
    "JaxLearningAssemblyResult",
    "LearningAssemblyError",
    "assemble_jax_learning_lifecycle",
]
