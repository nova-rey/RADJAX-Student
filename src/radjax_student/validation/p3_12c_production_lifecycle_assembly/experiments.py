"""Literal P3.12C assembly adversaries; each owns one public input mutation."""

from __future__ import annotations

from dataclasses import replace

from radjax_student.architecture import ArchitectureConfig, ArchitectureInitResult
from radjax_student.contracts import (
    ObjectiveConfig,
    ObjectiveIdentity,
)
from radjax_student.learning import assemble_jax_learning_lifecycle
from radjax_student.optimizers import OptimizerConfig, SgdOptimizer
from radjax_student.runtime import JaxRuntimeBackend, RuntimeConfig, RuntimeKeys
from radjax_student.validation.p3_11_9_replay.runner_jax import (
    StatefulLinearJaxArchitecture,
)

from .diagnostic import Invocation
from .fixtures import fresh_request_and_registries
from .implementation_audit import require_clean_synthetic_source


def _type_identity(value) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _assembly_input(request, registries) -> dict[str, object]:
    """Mechanical evidence only; it never receives case or expected metadata."""

    try:
        request_value: object = request.to_dict()
    except Exception:
        request_value = _type_identity(request)
    return {
        "request": request_value,
        "architecture_registry": sorted(
            (name, _type_identity(value))
            for name, value in registries.architecture_registry._plugins.items()
        ),
        "objective_registry": sorted(
            (str(name), _type_identity(value))
            for name, value in registries.objective_registry._plugins.items()
        ),
        "objective_descriptor_callable": getattr(
            registries.objective_registry.execution_descriptor, "__qualname__", ""
        ),
        "optimizer_registry": sorted(
            (name, _type_identity(value))
            for name, value in registries.optimizer_registry._backends.items()
        ),
        "runtime_registry": sorted(
            (name, _type_identity(value))
            for name, value in registries.runtime_registry._backends.items()
        ),
    }


def _assembly_invocation(request, registries) -> Invocation:
    baseline_request, baseline_registries = fresh_request_and_registries()
    return Invocation(
        assemble_jax_learning_lifecycle,
        (request,),
        {"registries": registries},
        _assembly_input(baseline_request, baseline_registries),
        _assembly_input(request, registries),
    )


def _audit_invocation(mutated_source: str) -> Invocation:
    return Invocation(
        require_clean_synthetic_source,
        (mutated_source,),
        {},
        "accepted source has no P3.12C authority defect",
        mutated_source,
    )


class _IncompleteInitArchitecture(StatefulLinearJaxArchitecture):
    def initialize_parameters(self, request):
        result = super().initialize_parameters(request)
        return ArchitectureInitResult(parameter_catalog=result.parameter_catalog)


class _InconsistentHFArchitecture(StatefulLinearJaxArchitecture):
    def initialize_parameters(self, request):
        result = super().initialize_parameters(request)
        object.__setattr__(result, "hf_reference", None)
        return result


class _UnsupportedSurfaceArchitecture(StatefulLinearJaxArchitecture):
    def resolve_objective_scope(self, scope, metadata):
        del scope
        del metadata
        return object()


class _WrongStateOptimizer(SgdOptimizer):
    def initialize_jax_state(self, **kwargs):
        del kwargs
        return object()


class _WrongIdentityOptimizer(SgdOptimizer):
    def initialize_jax_state(self, **kwargs):
        state = super().initialize_jax_state(**kwargs)
        object.__setattr__(state.envelope, "optimizer_id", "other.optimizer")
        return state


class _ContextBackendMismatchRuntime(JaxRuntimeBackend):
    def initialize_portability_context(self, *args, **kwargs):
        context = super().initialize_portability_context(*args, **kwargs)
        object.__setattr__(context, "backend_id", "other.runtime")
        return context


class _ContextSeedMismatchRuntime(JaxRuntimeBackend):
    def initialize_portability_context(self, *args, **kwargs):
        context = super().initialize_portability_context(*args, **kwargs)
        object.__setattr__(context, "root_seed", context.root_seed + 1)
        return context


class _KeyStreamMismatchRuntime(JaxRuntimeBackend):
    def initialize_portability_context(self, *args, **kwargs):
        context = super().initialize_portability_context(*args, **kwargs)
        object.__setattr__(
            context, "runtime_keys", RuntimeKeys.from_seed(context.root_seed + 1)
        )
        return context


def experiment_wrong_request_type() -> Invocation:
    _, registries = fresh_request_and_registries()
    return Invocation(
        assemble_jax_learning_lifecycle,
        (object(),),
        {"registries": registries},
        {"request_type": "JaxLearningAssemblyRequest"},
        {"request_type": "object"},
    )


def experiment_empty_architecture_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    object.__setattr__(request, "architecture_id", "")
    return _assembly_invocation(request, registries)


def experiment_empty_objective_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    object.__setattr__(request, "objective_identity", "")
    return _assembly_invocation(request, registries)


def experiment_empty_optimizer_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    object.__setattr__(request, "optimizer_id", "")
    return _assembly_invocation(request, registries)


def experiment_empty_runtime_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    object.__setattr__(request, "runtime_backend_id", "")
    return _assembly_invocation(request, registries)


def experiment_executable_plugin_injection_rejected() -> Invocation:
    request, registries = fresh_request_and_registries()
    object.__setattr__(request, "architecture_config", registries.architecture_registry)
    return _assembly_invocation(request, registries)


def experiment_unknown_architecture_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    config = ArchitectureConfig(
        "missing.architecture.v1", vocab_size=8, dtype_intent="float32"
    )
    return _assembly_invocation(
        replace(
            request, architecture_id=config.architecture_id, architecture_config=config
        ),
        registries,
    )


def experiment_architecture_config_identity_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    object.__setattr__(
        request,
        "architecture_config",
        ArchitectureConfig(
            "mismatch.architecture.v1", vocab_size=8, dtype_intent="float32"
        ),
    )
    return _assembly_invocation(request, registries)


def experiment_architecture_version_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    return _assembly_invocation(replace(request, architecture_version=2), registries)


def experiment_architecture_missing_jax_capability() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.architecture_registry._plugins[request.architecture_id] = object()
    return _assembly_invocation(request, registries)


def experiment_architecture_initialization_incomplete() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.architecture_registry._plugins[request.architecture_id] = (
        _IncompleteInitArchitecture(request.architecture_id)
    )
    return _assembly_invocation(request, registries)


def experiment_architecture_hf_descriptor_inconsistent() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.architecture_registry._plugins[request.architecture_id] = (
        _InconsistentHFArchitecture(request.architecture_id)
    )
    return _assembly_invocation(request, registries)


def experiment_unknown_objective_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    identity = ObjectiveIdentity("missing.objective", "1")
    config = ObjectiveConfig(identity, {"reduction": "mean"})
    return _assembly_invocation(
        replace(request, objective_identity=identity, objective_config=config),
        registries,
    )


def experiment_objective_config_identity_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    other = ObjectiveIdentity("other.objective", "1")
    object.__setattr__(
        request, "objective_config", ObjectiveConfig(other, {"reduction": "mean"})
    )
    return _assembly_invocation(request, registries)


def experiment_objective_version_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    identity = ObjectiveIdentity(request.objective_identity.objective_id, "2")
    return _assembly_invocation(
        replace(
            request,
            objective_identity=identity,
            objective_config=ObjectiveConfig(identity, {"reduction": "mean"}),
        ),
        registries,
    )


def experiment_objective_missing_jax_capability() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.objective_registry._plugins[request.objective_identity] = object()
    return _assembly_invocation(request, registries)


def experiment_objective_surface_unsupported() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.architecture_registry._plugins[request.architecture_id] = (
        _UnsupportedSurfaceArchitecture(request.architecture_id)
    )
    return _assembly_invocation(request, registries)


def experiment_objective_surface_not_architecture_derived() -> Invocation:
    return _audit_invocation("resolved_surface = caller_value")


def experiment_objective_descriptor_independently_fabricated() -> Invocation:
    request, registries = fresh_request_and_registries()

    def fabricated_descriptor(**kwargs):
        del kwargs
        return object()

    registries.objective_registry.execution_descriptor = fabricated_descriptor
    return _assembly_invocation(request, registries)


def experiment_unknown_optimizer_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    config = OptimizerConfig("missing.optimizer", learning_rate=0.25)
    return _assembly_invocation(
        replace(request, optimizer_id=config.optimizer_id, optimizer_config=config),
        registries,
    )


def experiment_optimizer_config_identity_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    object.__setattr__(
        request, "optimizer_config", OptimizerConfig("other", learning_rate=0.25)
    )
    return _assembly_invocation(request, registries)


def experiment_optimizer_missing_jax_capability() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.optimizer_registry._backends[request.optimizer_id] = object()
    return _assembly_invocation(request, registries)


def experiment_optimizer_state_identity_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.optimizer_registry._backends[request.optimizer_id] = (
        _WrongIdentityOptimizer()
    )
    return _assembly_invocation(request, registries)


def experiment_optimizer_state_not_backend_initialized() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.optimizer_registry._backends[request.optimizer_id] = (
        _WrongStateOptimizer()
    )
    return _assembly_invocation(request, registries)


def experiment_unknown_runtime_identity() -> Invocation:
    request, registries = fresh_request_and_registries()
    config = RuntimeConfig(
        backend_id="missing.runtime",
        platform_preference="cpu",
        precision_policy="float32",
        placement_policy="single_device",
        compilation_policy="eager",
        distributed_policy="disabled",
        fallback_policy="disallowed",
        seed=17,
    )
    return _assembly_invocation(
        replace(request, runtime_backend_id="missing.runtime", runtime_config=config),
        registries,
    )


def experiment_runtime_context_backend_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.runtime_registry._backends[request.runtime_backend_id] = (
        _ContextBackendMismatchRuntime()
    )
    return _assembly_invocation(request, registries)


def experiment_runtime_context_root_seed_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.runtime_registry._backends[request.runtime_backend_id] = (
        _ContextSeedMismatchRuntime()
    )
    return _assembly_invocation(request, registries)


def experiment_runtime_key_stream_root_seed_mismatch() -> Invocation:
    request, registries = fresh_request_and_registries()
    registries.runtime_registry._backends[request.runtime_backend_id] = (
        _KeyStreamMismatchRuntime()
    )
    return _assembly_invocation(request, registries)


def experiment_runtime_key_stream_not_backend_derived() -> Invocation:
    return _audit_invocation("runtime_key_stream = caller_value")


def experiment_lifecycle_component_replacement() -> Invocation:
    return _audit_invocation("lifecycle.architecture = other")


def experiment_loop_executor_architecture_replacement() -> Invocation:
    return _audit_invocation("loop.architecture = other")


def experiment_loop_executor_optimizer_replacement() -> Invocation:
    return _audit_invocation("loop.optimizer = other")


def experiment_loop_executor_objective_replacement() -> Invocation:
    return _audit_invocation("loop.objective = other")


def experiment_validation_manual_happy_path_detected() -> Invocation:
    return _audit_invocation("JaxLearningLifecycle()")


def experiment_production_imports_validation_assembly() -> Invocation:
    return _audit_invocation("from radjax_student.validation import assembly")


def experiment_competing_production_assembler_detected() -> Invocation:
    return _audit_invocation("def assemble_other(): pass")
