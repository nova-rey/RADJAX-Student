"""Backend-neutral loop adapter for the production JAX learning step."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import (
    ArchitectureConfig,
    ArchitectureState,
    JaxArchitecturePlugin,
    ParameterCatalog,
)
from radjax_student.architecture.models import _validate_hf_descriptor_against_init
from radjax_student.checkpoints import (
    JaxLearningCheckpointV3,
    load_learning_checkpoint_v3,
)
from radjax_student.checkpoints.npz_codec import (
    describe_mapping_pytree,
    descriptor_digest,
)
from radjax_student.contracts import (
    HFCompatibilityDescriptor,
    HFContractError,
    HFPreservationReference,
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveExecutionDescriptor,
    ParameterTreeLayout,
    ResolvedObjectiveSelection,
)
from radjax_student.learning import LearningBatch, LearningState
from radjax_student.learning.jax_batch import JaxBatchMaterializer
from radjax_student.objectives import ObjectiveRegistrySelection
from radjax_student.optimizers import (
    JaxOptimizerBackend,
    JaxOptimizerState,
    OptimizerConfig,
)
from radjax_student.runtime import (
    ExecutionBackend,
    ExecutionContext,
    RuntimeKeyStream,
)
from radjax_student.steps.jax_step import (
    JaxLearningStepExecution,
    execute_jax_learning_step,
)


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


@dataclass(frozen=True)
class JaxLearningLifecycle:
    """Typed ownership carrier for one JAX learning run.

    Learning transports this object but does not interpret architecture carry or
    optimizer numerical leaves. The object is replaced after every execution.
    """

    architecture: JaxArchitecturePlugin
    architecture_config: ArchitectureConfig
    architecture_state: ArchitectureState | None
    architecture_carry: Any
    parameter_catalog: ParameterCatalog
    parameter_layout: ParameterTreeLayout
    hf_descriptor: HFCompatibilityDescriptor
    hf_reference: HFPreservationReference
    objective_selection: ObjectiveRegistrySelection
    objective_config: ObjectiveConfig
    resolved_objective_selection: ResolvedObjectiveSelection
    objective_descriptor: ObjectiveExecutionDescriptor
    optimizer: JaxOptimizerBackend
    optimizer_config: OptimizerConfig
    optimizer_state: JaxOptimizerState
    parameters: Any
    learning_state: LearningState
    runtime_context: ExecutionContext
    runtime_backend: ExecutionBackend
    runtime_key_stream: RuntimeKeyStream
    architecture_carry_descriptor: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, JaxArchitecturePlugin):
            raise TypeError("lifecycle requires a complete JaxArchitecturePlugin")
        if not isinstance(self.architecture_config, ArchitectureConfig):
            raise TypeError("architecture_config must be ArchitectureConfig")
        if (
            self.architecture_config.architecture_id
            != self.architecture.architecture_id
        ):
            raise ValueError("architecture config identity does not match plugin")
        if self.architecture_state is not None and not isinstance(
            self.architecture_state, ArchitectureState
        ):
            raise TypeError("architecture_state must be ArchitectureState when set")
        if not isinstance(self.parameter_catalog, ParameterCatalog):
            raise TypeError("parameter_catalog must be ParameterCatalog")
        if self.parameter_catalog.architecture_id != self.architecture.architecture_id:
            raise ValueError("parameter catalog identity does not match plugin")
        if not isinstance(self.parameter_layout, ParameterTreeLayout):
            raise TypeError("parameter_layout must be ParameterTreeLayout")
        if self.parameter_layout.architecture_id != self.architecture.architecture_id:
            raise ValueError("parameter layout identity does not match plugin")
        if set(self.parameter_catalog.paths) != set(
            self.parameter_layout.logical_paths
        ):
            raise ValueError("parameter catalog and layout paths must match")
        if not isinstance(self.hf_descriptor, HFCompatibilityDescriptor):
            raise TypeError("hf_descriptor must be HFCompatibilityDescriptor")
        if not isinstance(self.hf_reference, HFPreservationReference):
            raise TypeError("hf_reference must be HFPreservationReference")
        _validate_hf_descriptor_against_init(
            self.hf_descriptor,
            parameter_catalog=self.parameter_catalog,
            parameter_layout=self.parameter_layout,
            architecture_config=self.architecture_config,
            architecture_version=self.architecture.architecture_version,
        )
        if self.hf_reference != self.hf_descriptor.preservation_reference():
            raise HFContractError(
                "hf_reference_derivation_mismatch",
                "HF reference was not derived from the lifecycle descriptor",
            )
        if not isinstance(self.objective_selection, ObjectiveRegistrySelection):
            raise TypeError("lifecycle requires ObjectiveRegistrySelection")
        if not self.objective_selection.is_registry_selected:
            raise ObjectiveContractError(
                "objective_identity_mismatch",
                "lifecycle objective must be selected through ObjectiveRegistry",
            )
        if not isinstance(self.objective_config, ObjectiveConfig):
            raise TypeError("lifecycle requires ObjectiveConfig")
        if not isinstance(
            self.resolved_objective_selection, ResolvedObjectiveSelection
        ):
            raise TypeError("lifecycle requires ResolvedObjectiveSelection")
        if not isinstance(self.objective_descriptor, ObjectiveExecutionDescriptor):
            raise TypeError("lifecycle requires ObjectiveExecutionDescriptor")
        objective = self.objective_selection
        if (
            self.objective_config.identity != objective.identity
            or self.objective_descriptor.identity != objective.identity
            or self.objective_descriptor.capability_profile_digest
            != objective.profile.digest
            or self.objective_descriptor.config_digest != self.objective_config.digest
            or self.objective_descriptor.resolved_surface_identity
            != self.resolved_objective_selection.digest
            or self.objective_descriptor.metric_schema_id
            != objective.profile.metric_schema_id
            or self.objective_descriptor.implementation_identity
            != objective.implementation_identity
        ):
            raise ObjectiveContractError(
                "objective_config_identity_mismatch",
                "lifecycle objective identity is inconsistent",
            )
        objective.plugin.validate_config(self.objective_config)
        objective.plugin.validate_resolved_surface(self.resolved_objective_selection)
        architecture_selection = self.architecture.resolve_objective_scope(
            self.learning_state.active_objective_scope,
            self.architecture.architecture_metadata(),
        )
        if architecture_selection != self.resolved_objective_selection:
            raise ObjectiveContractError(
                "objective_surface_identity_mismatch",
                "lifecycle objective surface was not resolved by its architecture",
            )
        if not isinstance(self.optimizer, JaxOptimizerBackend):
            raise TypeError("lifecycle requires a complete JaxOptimizerBackend")
        if self.optimizer_config.optimizer_id != self.optimizer.optimizer_id:
            raise ValueError("optimizer config identity does not match plugin")
        if not isinstance(self.optimizer_state, JaxOptimizerState):
            raise TypeError("optimizer_state must be JaxOptimizerState")
        if self.optimizer_state.envelope.optimizer_id != self.optimizer.optimizer_id:
            raise ValueError("optimizer state identity does not match plugin")
        if not isinstance(self.learning_state, LearningState):
            raise TypeError("learning_state must be LearningState")
        if not isinstance(self.runtime_context, ExecutionContext):
            raise TypeError("runtime_context must be ExecutionContext")
        if self.runtime_context.backend_id != self.runtime_backend.backend_id:
            raise ValueError("runtime context identity does not match backend")
        if not isinstance(self.runtime_key_stream, RuntimeKeyStream):
            raise TypeError("runtime_key_stream must be RuntimeKeyStream")
        if self.runtime_key_stream.root_seed != self.runtime_context.root_seed:
            raise ValueError("runtime key stream does not belong to the runtime")
        if not isinstance(self.architecture_carry_descriptor, Mapping):
            raise TypeError(
                "architecture_carry_descriptor must be supplied by initialization"
            )
        carry_identity = dict(self.architecture_carry_descriptor)
        expected_descriptor_digest = descriptor_digest(
            describe_mapping_pytree(self.architecture_carry)
        )
        if carry_identity.get("pytree_descriptor_digest") != expected_descriptor_digest:
            raise ValueError(
                "architecture carry descriptor does not match initialized carry"
            )
        if self.architecture_state is not None and carry_identity.get(
            "state_id"
        ) not in (
            None,
            self.architecture_state.state_id,
        ):
            raise ValueError(
                "architecture carry descriptor state does not match lifecycle"
            )
        object.__setattr__(
            self,
            "architecture_carry_descriptor",
            MappingProxyType(carry_identity),
        )

    @property
    def config_digest(self) -> str:
        return _digest(self.architecture_config.to_dict())

    @property
    def catalog_digest(self) -> str:
        return _digest(self.parameter_catalog.to_dict())

    @property
    def runtime_reference(self) -> str:
        return self.runtime_context.runtime_id

    def checkpoint(self) -> JaxLearningCheckpointV3:
        """Expose a validated continuation checkpoint without array conversion."""

        return JaxLearningCheckpointV3(
            runtime_reference=self.runtime_reference,
            learning_state=self.learning_state,
            optimizer_state=self.optimizer_state,
            parameters=self.parameters,
            architecture_carry=self.architecture_carry,
            parameter_layout=self.parameter_layout,
            architecture_state=self.architecture_state,
            hf_descriptor=self.hf_descriptor,
            hf_reference=self.hf_reference,
            objective_config=self.objective_config,
            resolved_objective_selection=self.resolved_objective_selection,
            objective_descriptor=self.objective_descriptor,
            objective_registry_selection=self.objective_selection.to_dict(),
            architecture_config_digest=self.config_digest,
            parameter_catalog_digest=self.catalog_digest,
            architecture_carry_descriptor=self.architecture_carry_descriptor,
        )

    def with_checkpoint(
        self, checkpoint: JaxLearningCheckpointV3
    ) -> JaxLearningLifecycle:
        """Return this lifecycle restored from a caller-validated v3 checkpoint."""

        if checkpoint.runtime_reference != self.runtime_reference:
            raise ValueError("checkpoint runtime identity does not match lifecycle")
        if checkpoint.parameter_layout.digest() != self.parameter_layout.digest():
            raise ValueError("checkpoint parameter layout does not match lifecycle")
        if checkpoint.hf_descriptor != self.hf_descriptor:
            raise ValueError("checkpoint HF descriptor does not match lifecycle")
        if checkpoint.hf_reference != self.hf_reference:
            raise ValueError("checkpoint HF identity does not match lifecycle")
        if checkpoint.architecture_config_digest != self.config_digest:
            raise ValueError("checkpoint config identity does not match lifecycle")
        if checkpoint.parameter_catalog_digest != self.catalog_digest:
            raise ValueError("checkpoint catalog identity does not match lifecycle")
        if checkpoint.objective_descriptor != self.objective_descriptor:
            raise ValueError("checkpoint objective identity does not match lifecycle")
        if checkpoint.objective_config != self.objective_config:
            raise ValueError("checkpoint objective config does not match lifecycle")
        if checkpoint.resolved_objective_selection != self.resolved_objective_selection:
            raise ValueError("checkpoint objective surface does not match lifecycle")
        return replace(
            self,
            learning_state=checkpoint.learning_state,
            optimizer_state=checkpoint.optimizer_state,
            parameters=checkpoint.parameters,
            architecture_carry=checkpoint.architecture_carry,
            architecture_state=checkpoint.architecture_state,
            architecture_carry_descriptor=checkpoint.architecture_carry_descriptor,
        )

    def restore_from_checkpoint(self, directory: Any) -> JaxLearningLifecycle:
        """Restore only through the lifecycle's caller-bound identity contract."""

        checkpoint = load_learning_checkpoint_v3(
            directory,
            optimizer=self.optimizer,
            parameter_layout=self.parameter_layout,
            runtime_reference=self.runtime_reference,
            expected_hf_reference=self.hf_reference,
            expected_hf_descriptor=self.hf_descriptor,
            expected_architecture_config_digest=self.config_digest,
            expected_parameter_catalog_digest=self.catalog_digest,
            expected_architecture_state_id=(
                None
                if self.architecture_state is None
                else self.architecture_state.state_id
            ),
            expected_architecture_carry_descriptor=self.architecture_carry_descriptor,
            expected_objective_descriptor=self.objective_descriptor,
            expected_objective_config=self.objective_config,
            expected_resolved_objective_selection=self.resolved_objective_selection,
            expected_objective_selection=self.objective_selection,
            require_objective_identity=True,
        )
        return self.with_checkpoint(checkpoint)


@dataclass
class JaxLoopExecutor:
    """Callable generic-loop executor that advances a lifecycle functionally."""

    lifecycle: JaxLearningLifecycle
    batch_materializer: JaxBatchMaterializer
    execution_request_factory: Any
    precision_policy: str | None = None
    schedule_values: Mapping[str, Any] | None = None
    rng_slot: str = "dropout"

    def __post_init__(self) -> None:
        if not isinstance(self.batch_materializer, JaxBatchMaterializer):
            raise TypeError("batch_materializer must implement JaxBatchMaterializer")
        if not callable(self.execution_request_factory):
            raise TypeError("execution_request_factory must be callable")
        if not isinstance(self.rng_slot, str) or not self.rng_slot:
            raise ValueError("rng_slot must be nonempty")
        self.schedule_values = MappingProxyType(dict(self.schedule_values or {}))

    def __call__(self, **kwargs: Any) -> JaxLearningStepExecution:
        lifecycle = self.lifecycle
        if kwargs["architecture"] is not lifecycle.architecture:
            raise ValueError("loop architecture does not match JAX lifecycle")
        if kwargs["optimizer"] is not lifecycle.optimizer:
            raise ValueError("loop optimizer does not match JAX lifecycle")
        if kwargs["architecture_config"] != lifecycle.architecture_config:
            raise ValueError("loop architecture config does not match JAX lifecycle")
        if kwargs["optimizer_config"] != lifecycle.optimizer_config:
            raise ValueError("loop optimizer config does not match JAX lifecycle")
        if kwargs["learning_state"] != lifecycle.learning_state:
            raise ValueError("loop learning state does not match JAX lifecycle")
        if kwargs["objective"] != lifecycle.objective_selection:
            raise ValueError("loop objective must be the lifecycle registry selection")
        # A checkpoint callback may replace this immutable lifecycle between
        # iterations. The generic loop deliberately keeps no JAX-specific
        # state, so this adapter remains the single owner of restored arrays.
        batch = kwargs["batch"]
        if not isinstance(batch, LearningBatch):
            raise TypeError("loop batch must be LearningBatch")
        execution = execute_jax_learning_step(
            architecture=lifecycle.architecture,
            architecture_config=lifecycle.architecture_config,
            objective_selection=lifecycle.objective_selection,
            objective_config=lifecycle.objective_config,
            objective_descriptor=lifecycle.objective_descriptor,
            resolved_objective_selection=lifecycle.resolved_objective_selection,
            optimizer=lifecycle.optimizer,
            optimizer_config=lifecycle.optimizer_config,
            optimizer_state=lifecycle.optimizer_state,
            learning_state=lifecycle.learning_state,
            parameters=lifecycle.parameters,
            architecture_carry=lifecycle.architecture_carry,
            learning_batch=batch,
            batch_materializer=self.batch_materializer,
            parameter_layout=lifecycle.parameter_layout,
            runtime_key_stream=lifecycle.runtime_key_stream,
            rng_slot=self.rng_slot,
            rng_invocation_index=lifecycle.learning_state.global_step,
            runtime_context=lifecycle.runtime_context,
            runtime_backend=lifecycle.runtime_backend,
            execution_request=self.execution_request_factory(lifecycle.learning_state),
            precision_policy=self.precision_policy,
            schedule_values=self.schedule_values,
        )
        return self.accept_execution(execution)

    def accept_execution(
        self, execution: JaxLearningStepExecution
    ) -> JaxLearningStepExecution:
        """Accept only complete production JAX executions into this lifecycle."""

        if not isinstance(execution, JaxLearningStepExecution):
            raise TypeError("JAX lifecycle rejects legacy or partial step executions")
        lifecycle = self.lifecycle
        self.lifecycle = replace(
            lifecycle,
            learning_state=execution.learning_state,
            optimizer_state=execution.optimizer_state,
            parameters=execution.parameters,
            architecture_carry=execution.architecture_carry,
        )
        return execution


__all__ = ["JaxLearningLifecycle", "JaxLoopExecutor"]
