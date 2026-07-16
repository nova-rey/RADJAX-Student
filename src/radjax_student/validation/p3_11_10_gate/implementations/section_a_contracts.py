"""Literal Section A registry and contract experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from radjax_student.architecture import (
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureRegistry,
    ParameterCatalog,
)
from radjax_student.architecture.testing import (
    FAKE_ARCHITECTURE_CAPABILITIES,
    FakeArchitecturePlugin,
)
from radjax_student.contracts import ParameterTreeLayout, UpdateScope
from radjax_student.optimizers import (
    OptimizerCapabilityProfile,
    OptimizerRegistry,
    SgdOptimizer,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_memory_experiment,
    public_boundary,
)
from radjax_student.validation.p3_11_10_gate.implementations.literal_fixtures import (
    checkpoint_payload,
)


class ApplyJaxOnlyArchitecture:
    def apply_jax(self, *args: Any, **kwargs: Any) -> Any:
        return None


class ProfileIdMismatchArchitecture(FakeArchitecturePlugin):
    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return ArchitectureCapabilityProfile(
            "foreign.architecture.v1",
            self.architecture_version,
            FAKE_ARCHITECTURE_CAPABILITIES,
        )


class JaxDeclaredWithoutImplementationArchitecture(FakeArchitecturePlugin):
    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            self.architecture_version,
            (*FAKE_ARCHITECTURE_CAPABILITIES, "architecture.jax_execution_v1"),
        )


class JaxUndeclaredImplementationArchitecture(FakeArchitecturePlugin):
    def apply_jax(self, *args: Any, **kwargs: Any) -> Any:
        return None


class JaxHelpersOnlyOptimizer:
    optimizer_id = "helpers.only"
    optimizer_version = 1

    def jax_state_descriptor(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def initialize_jax_state(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def validate_jax_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def apply_jax_updates(self, *args: Any, **kwargs: Any) -> Any:
        return None


class JaxExecutionWithoutBackendIdentityOptimizer:
    """A distinct JAX-only object lacking the optimizer backend contract."""

    optimizer_id = "execution.without.backend"
    optimizer_version = 2

    def jax_state_descriptor(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def initialize_jax_state(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def validate_jax_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def apply_jax_updates(self, *args: Any, **kwargs: Any) -> Any:
        return None


class ProfileMismatchOptimizer(SgdOptimizer):
    def capability_profile(self) -> OptimizerCapabilityProfile:
        return OptimizerCapabilityProfile(
            "foreign.optimizer",
            self.optimizer_version,
            super().capability_profile().capabilities,
        )


class FullDeclaredJaxMissingOptimizer:
    optimizer_id = "declared.no.jax"
    optimizer_version = 1

    def __init__(self) -> None:
        self._inner = SgdOptimizer()

    def capability_profile(self) -> OptimizerCapabilityProfile:
        return OptimizerCapabilityProfile(
            self.optimizer_id,
            self.optimizer_version,
            (*self._inner.capability_profile().capabilities,),
        )

    def validate_config(self, config: Any) -> None:
        self._inner.validate_config(config)

    def initialize_state(self, request: Any) -> Any:
        return self._inner.initialize_state(request)

    def apply_updates(self, request: Any) -> Any:
        return self._inner.apply_updates(request)

    def describe_state(self, state: Any) -> Any:
        return self._inner.describe_state(state)


class RawParameterObjective:
    def evaluate(self, surface: Any, targets: Any, weights: Any, config: Any) -> Any:
        del targets, weights, config
        return surface["parameters"]


class LegacyStudentRegistryProtocol:
    backend_id = "legacy.student"

    def register_backend(self, backend: Any) -> None:
        del backend


@public_boundary("registry_validation")
def _register_architecture(plugin: Any) -> tuple[str, ...]:
    registry = ArchitectureRegistry()
    registry.register(plugin)
    return registry.list_plugins()


@public_boundary("registry_validation")
def _register_architecture_with_mismatched_key(
    payload: tuple[Any, str],
) -> tuple[str, ...]:
    plugin, registry_id = payload
    registry = ArchitectureRegistry()
    registry.register(plugin, registry_id=registry_id)
    return registry.list_plugins()


@public_boundary("registry_validation")
def _register_optimizer(backend: Any) -> tuple[str, ...]:
    registry = OptimizerRegistry()
    registry.register(backend)
    return registry.list_optimizers()


@public_boundary("registry_validation")
def _register_optimizer_with_mismatched_key(
    payload: tuple[Any, str],
) -> tuple[str, ...]:
    backend, registry_id = payload
    registry = OptimizerRegistry()
    registry.register(backend, registry_id=registry_id)
    return registry.list_optimizers()


@public_boundary("registry_validation")
def _validate_architecture_config(
    payload: tuple[FakeArchitecturePlugin, ArchitectureConfig],
) -> None:
    plugin, config = payload
    plugin.validate_config(config)


@public_boundary("registry_validation")
def _validate_catalog(payload: tuple[FakeArchitecturePlugin, ParameterCatalog]) -> None:
    plugin, catalog = payload
    plugin.resolve_update_scope(UpdateScope(), catalog)


@public_boundary("registry_validation")
def _validate_layout_identity(payload: tuple[ParameterTreeLayout, str]) -> str:
    layout, architecture_id = payload
    if layout.architecture_id != architecture_id:
        raise ValueError("parameter layout identity does not match architecture")
    return layout.architecture_id


@public_boundary("registry_validation")
def _validate_hf_identity(payload: tuple[Any, ParameterTreeLayout]) -> str:
    hf_reference, layout = payload
    if hf_reference.architecture_id != layout.architecture_id:
        raise ValueError("HF reference architecture identity does not match layout")
    return hf_reference.architecture_id


@public_boundary("registry_validation")
def _evaluate_objective_surface(payload: tuple[RawParameterObjective, Any]) -> Any:
    objective, surface = payload
    return objective.evaluate(surface, {}, {}, object())


@public_boundary("registry_validation")
def _reject_legacy_registry(payload: LegacyStudentRegistryProtocol) -> tuple[str, ...]:
    registry = ArchitectureRegistry()
    registry.register(payload)  # type: ignore[arg-type]
    return registry.list_plugins()


@public_boundary("registry_validation")
def _reject_legacy_step_result(payload: Any) -> None:
    from radjax_student.steps.jax_loop import JaxLoopExecutor

    JaxLoopExecutor(payload, payload, payload, payload)  # type: ignore[arg-type]


def _record(
    context: GateExecutionContext,
    baseline: Any,
    mutated: Any,
    path: str,
    operation: str,
    callable_: Any,
    baseline_callable: Any | None = None,
) -> ExperimentExecution:
    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="architecture_or_optimizer_contract",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=callable_,
        baseline_callable=callable_ if baseline_callable is None else baseline_callable,
    )


def experiment_a_complete_architecture_and_optimizer_register_and_execute(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = (FakeArchitecturePlugin(), SgdOptimizer())
    mutated = (
        FakeArchitecturePlugin(architecture_id="test.architecture.control"),
        SgdOptimizer(),
    )

    @public_boundary("registry_validation")
    def register_pair(
        value: tuple[Any, Any],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        architecture, optimizer = value
        architectures = ArchitectureRegistry()
        optimizers = OptimizerRegistry()
        architectures.register(architecture)
        optimizers.register(optimizer)
        return architectures.list_plugins(), optimizers.list_optimizers()

    return _record(
        context,
        baseline,
        mutated,
        "registry.pair",
        "register_complete_plugins",
        register_pair,
    )


def experiment_a_architecture_apply_jax_only(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = FakeArchitecturePlugin()
    mutated = ApplyJaxOnlyArchitecture()
    return _record(
        context,
        baseline,
        mutated,
        "architecture.apply_jax",
        "remove_complete_architecture_contract",
        _register_architecture,
    )


def experiment_a_architecture_registry_id_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = (FakeArchitecturePlugin(), "test.architecture.v1")
    mutated = (FakeArchitecturePlugin(), "foreign.architecture.v1")
    return _record(
        context,
        baseline,
        mutated,
        "registry_id",
        "replace_registry_id",
        _register_architecture_with_mismatched_key,
    )


def experiment_a_architecture_capability_profile_id_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = FakeArchitecturePlugin()
    mutated = ProfileIdMismatchArchitecture()
    return _record(
        context,
        baseline,
        mutated,
        "capability_profile.architecture_id",
        "replace_capability_profile_identity",
        _register_architecture,
    )


def experiment_a_architecture_declared_jax_missing_implementation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = FakeArchitecturePlugin()
    mutated = JaxDeclaredWithoutImplementationArchitecture()
    return _record(
        context,
        baseline,
        mutated,
        "capability_profile.capabilities",
        "declare_jax_without_apply_jax",
        _register_architecture,
    )


def experiment_a_architecture_undeclared_jax_implementation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = FakeArchitecturePlugin()
    mutated = JaxUndeclaredImplementationArchitecture()
    return _record(
        context,
        baseline,
        mutated,
        "apply_jax",
        "add_undeclared_jax_execution",
        _register_architecture,
    )


def experiment_a_architecture_config_different_id(
    context: GateExecutionContext,
) -> ExperimentExecution:
    plugin = FakeArchitecturePlugin()
    baseline = (plugin, ArchitectureConfig(plugin.architecture_id))
    mutated = (plugin, ArchitectureConfig("foreign.architecture.v1"))
    return _record(
        context,
        baseline,
        mutated,
        "architecture_config.architecture_id",
        "replace_configuration_architecture_id",
        _validate_architecture_config,
    )


def experiment_a_parameter_catalog_different_architecture_id(
    context: GateExecutionContext,
) -> ExperimentExecution:
    plugin = FakeArchitecturePlugin()
    baseline = (plugin, plugin.describe_parameters())
    catalog = plugin.describe_parameters()
    mutated = (plugin, replace(catalog, architecture_id="foreign.architecture.v1"))
    return _record(
        context,
        baseline,
        mutated,
        "parameter_catalog.architecture_id",
        "replace_catalog_architecture_id",
        _validate_catalog,
    )


def experiment_a_parameter_layout_different_architecture_id(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = checkpoint_payload(SgdOptimizer())
    foreign_layout = replace(
        baseline.parameter_layout, architecture_id="foreign.architecture.v1"
    )
    baseline_input = (
        baseline.parameter_layout,
        baseline.parameter_layout.architecture_id,
    )
    mutated = (foreign_layout, baseline.parameter_layout.architecture_id)
    return _record(
        context,
        baseline_input,
        mutated,
        "parameter_layout.architecture_id",
        "replace_layout_architecture_id",
        _validate_layout_identity,
    )


def experiment_a_hf_reference_different_architecture_id(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = checkpoint_payload(SgdOptimizer())
    hf = replace(baseline.hf_reference, architecture_id="foreign.architecture.v1")
    baseline_input = (baseline.hf_reference, baseline.parameter_layout)
    mutated = (hf, baseline.parameter_layout)
    return _record(
        context,
        baseline_input,
        mutated,
        "hf_reference.architecture_id",
        "replace_hf_architecture_id",
        _validate_hf_identity,
    )


def experiment_a_optimizer_jax_helpers_only(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = SgdOptimizer()
    mutated = JaxHelpersOnlyOptimizer()
    return _record(
        context,
        baseline,
        mutated,
        "optimizer.jax_helpers",
        "remove_optimizer_backend_contract",
        _register_optimizer,
    )


def experiment_a_optimizer_registry_id_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = (SgdOptimizer(), "sgd.v1")
    mutated = (SgdOptimizer(), "foreign.optimizer")
    return _record(
        context,
        baseline,
        mutated,
        "registry_id",
        "replace_optimizer_registry_id",
        _register_optimizer_with_mismatched_key,
    )


def experiment_a_optimizer_capability_profile_id_or_version_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = SgdOptimizer()
    mutated = ProfileMismatchOptimizer()
    return _record(
        context,
        baseline,
        mutated,
        "optimizer.capability_profile.optimizer_id",
        "replace_optimizer_capability_identity",
        _register_optimizer,
    )


def experiment_a_optimizer_declared_jax_missing_implementation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = SgdOptimizer()
    mutated = FullDeclaredJaxMissingOptimizer()
    return _record(
        context,
        baseline,
        mutated,
        "optimizer.capability_profile.capabilities",
        "declare_optimizer_jax_without_methods",
        _register_optimizer,
    )


def experiment_a_optimizer_jax_implementation_missing_backend_identity(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = SgdOptimizer()
    mutated = JaxExecutionWithoutBackendIdentityOptimizer()
    return _record(
        context,
        baseline,
        mutated,
        "optimizer.backend_identity",
        "remove_full_optimizer_backend_identity",
        _register_optimizer,
    )


def experiment_a_objective_raw_parameter_tree_assumption(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = (RawParameterObjective(), {"parameters": 1.0})
    mutated = (RawParameterObjective(), 1.0)
    return _record(
        context,
        baseline,
        mutated,
        "objective.surface",
        "replace_surface_with_parameterless_scalar",
        _evaluate_objective_surface,
    )


def experiment_a_legacy_student_registry_protocol(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = FakeArchitecturePlugin()
    mutated = LegacyStudentRegistryProtocol()
    return _record(
        context,
        baseline,
        mutated,
        "legacy.student_registry",
        "submit_legacy_registry_protocol",
        _reject_legacy_registry,
    )


def experiment_a_legacy_jax_step_into_loop_executor(
    context: GateExecutionContext,
) -> ExperimentExecution:
    from radjax_student.legacy.scalar_learning import LegacyScalarStepExecution
    from radjax_student.steps.jax_loop import JaxLoopExecutor

    baseline = {"execution_type": "JaxLearningStepExecution"}
    mutated = LegacyScalarStepExecution(None, None, None, {})

    @public_boundary("registry_validation")
    def accept_legacy(value: Any) -> None:
        # ``accept_execution`` checks its concrete argument before reading the
        # lifecycle, so the uninitialized adapter still exercises the real
        # production legacy-rejection boundary.
        JaxLoopExecutor.accept_execution(object.__new__(JaxLoopExecutor), value)

    return _record(
        context,
        baseline,
        mutated,
        "legacy.scalar_step_execution",
        "submit_legacy_step_result_to_jax_executor",
        accept_legacy,
        baseline_callable=lambda value: value,
    )


SECTION_IMPLEMENTATIONS = {
    "A.positive.complete_architecture_and_optimizer_register_and_execute": GateCaseImplementation(  # noqa: E501
        experiment_a_complete_architecture_and_optimizer_register_and_execute
    ),
    "A.reject.architecture_apply_jax_only": GateCaseImplementation(
        experiment_a_architecture_apply_jax_only
    ),
    "A.reject.architecture_registry_id_mismatch": GateCaseImplementation(
        experiment_a_architecture_registry_id_mismatch
    ),
    "A.reject.architecture_capability_profile_id_mismatch": GateCaseImplementation(
        experiment_a_architecture_capability_profile_id_mismatch
    ),
    "A.reject.architecture_declared_jax_missing_implementation": GateCaseImplementation(
        experiment_a_architecture_declared_jax_missing_implementation
    ),
    "A.reject.architecture_undeclared_jax_implementation": GateCaseImplementation(
        experiment_a_architecture_undeclared_jax_implementation
    ),
    "A.reject.architecture_config_different_id": GateCaseImplementation(
        experiment_a_architecture_config_different_id
    ),
    "A.reject.parameter_catalog_different_architecture_id": GateCaseImplementation(
        experiment_a_parameter_catalog_different_architecture_id
    ),
    "A.reject.parameter_layout_different_architecture_id": GateCaseImplementation(
        experiment_a_parameter_layout_different_architecture_id
    ),
    "A.reject.hf_reference_different_architecture_id": GateCaseImplementation(
        experiment_a_hf_reference_different_architecture_id
    ),
    "A.reject.optimizer_jax_helpers_only": GateCaseImplementation(
        experiment_a_optimizer_jax_helpers_only
    ),
    "A.reject.optimizer_registry_id_mismatch": GateCaseImplementation(
        experiment_a_optimizer_registry_id_mismatch
    ),
    "A.reject.optimizer_capability_profile_id_or_version_mismatch": GateCaseImplementation(  # noqa: E501
        experiment_a_optimizer_capability_profile_id_or_version_mismatch
    ),
    "A.reject.optimizer_declared_jax_missing_implementation": GateCaseImplementation(
        experiment_a_optimizer_declared_jax_missing_implementation
    ),
    "A.reject.optimizer_jax_implementation_missing_backend_identity": GateCaseImplementation(  # noqa: E501
        experiment_a_optimizer_jax_implementation_missing_backend_identity
    ),
    "A.reject.objective_raw_parameter_tree_assumption": GateCaseImplementation(
        experiment_a_objective_raw_parameter_tree_assumption
    ),
    "A.reject.legacy_student_registry_protocol": GateCaseImplementation(
        experiment_a_legacy_student_registry_protocol
    ),
    "A.reject.legacy_jax_step_into_loop_executor": GateCaseImplementation(
        experiment_a_legacy_jax_step_into_loop_executor
    ),
}


__all__ = ["SECTION_IMPLEMENTATIONS"]
