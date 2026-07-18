"""Complete runtime-hosted JAX learning step for the P3.11 conveyor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import ArchitectureConfig, JaxArchitecturePlugin
from radjax_student.contracts import (
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveExecutionDescriptor,
    ParameterTreeLayout,
    ResolvedObjectiveSelection,
)
from radjax_student.learning import (
    LearningBatch,
    LearningState,
    LearningStepResult,
    LossResult,
    MetricRecord,
)
from radjax_student.learning.jax_batch import JaxBatchMaterializer
from radjax_student.learning.jax_core import (
    JaxBatch,
    JaxLossAuxiliary,
    build_registered_jax_loss_fn,
    build_value_and_grad_fn,
    validate_finite_loss_and_gradients,
)
from radjax_student.learning.jax_execution import prepare_jax_execution_plan
from radjax_student.objectives import ObjectiveRegistrySelection
from radjax_student.optimizers import (
    JaxOptimizerBackend,
    JaxOptimizerState,
    OptimizerConfig,
    advanced_jax_optimizer_state,
    require_finite_jax_gradients,
    validate_jax_optimizer_state,
)
from radjax_student.runtime import (
    ExecutionBackend,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    RuntimeCallableBinding,
    execute_function,
)
from radjax_student.runtime.callables import (
    CALLABLE_DECLARATION_SCHEMA_VERSION,
    RuntimeCallableDeclaration,
)
from radjax_student.runtime.jax_bridge import (
    JAX_KEY_BRIDGE_VERSION,
    JAX_PRNG_IMPLEMENTATION,
    derive_jax_key,
    validate_runtime_jax_key_request,
)
from radjax_student.runtime.jax_inputs import prepare_jax_inputs
from radjax_student.runtime.keys import RuntimeKeys, RuntimeKeyStream

GENERIC_JAX_LEARNING_STEP_DECLARATION = RuntimeCallableDeclaration(
    schema_version=CALLABLE_DECLARATION_SCHEMA_VERSION,
    callable_id="radjax.learning.generic_jax_step",
    callable_version=1,
    owner="steps",
    implementation_module="radjax_student.steps.jax_step",
    implementation_qualname="execute_jax_learning_step_kernel",
    input_contract_id="radjax.jax_learning_step_kernel_input.v1",
    output_contract_id="radjax.jax_learning_step_kernel_output.v1",
    claims_not_made=("transitive_dependency_semantics_not_fully_hashed",),
)


def execute_jax_learning_step_kernel(
    kernel: Any,
    *args: Any,
) -> Any:
    """Named generic runtime target; orchestration supplies the JAX kernel body.

    P3.12D binds this production operation through runtime.  The callable body is
    deliberately small: objective and optimizer semantics remain with their
    existing owners and are never interpreted by runtime.
    """

    return kernel(*args)


class JaxUpdateEvidenceError(ValueError):
    """Stable public rejection for post-update logical-path evidence."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class JaxBatchBindingError(ValueError):
    """Stable public failure when a materialized batch is not its source batch."""

    code = "jax_batch_source_mismatch"


def validate_jax_update_evidence(
    *,
    before_parameters: Mapping[str, Any],
    after_parameters: Mapping[str, Any],
    parameter_layout: ParameterTreeLayout,
    selected_paths: tuple[str, ...],
    changed_paths: tuple[str, ...],
    unchanged_paths: tuple[str, ...],
    optimizer_before: Mapping[str, Any] | None = None,
    optimizer_after: Mapping[str, Any] | None = None,
) -> None:
    """Bind reported logical paths to actual parameter/state transitions.

    The optimizer owns numerical-state meaning, but this boundary verifies that
    an update report neither claims movement where no selected parameter moved
    nor hides movement in an excluded logical parameter or its supplied
    per-parameter optimizer-state view.
    """

    from importlib import import_module

    jnp = import_module("jax.numpy")
    parameter_layout.validate_materialized_parameters(before_parameters)
    parameter_layout.validate_materialized_parameters(after_parameters)
    selected = set(selected_paths)
    changed = set(changed_paths)
    unchanged = set(unchanged_paths)
    all_paths = set(parameter_layout.logical_paths)
    if selected - all_paths:
        raise JaxUpdateEvidenceError(
            "jax_update_selection_unknown_path", "selected paths are not in the layout"
        )
    if changed & unchanged:
        raise JaxUpdateEvidenceError(
            "jax_update_paths_overlap", "changed and unchanged path reports overlap"
        )
    if changed | unchanged != all_paths:
        raise JaxUpdateEvidenceError(
            "jax_update_paths_incomplete", "update report does not cover the layout"
        )
    for entry in parameter_layout.entries:
        before = _mapping_leaf(before_parameters, entry.jax_keypath)
        after = _mapping_leaf(after_parameters, entry.jax_keypath)
        moved = not bool(jnp.array_equal(jnp.asarray(before), jnp.asarray(after)))
        if entry.logical_path not in selected and moved:
            raise JaxUpdateEvidenceError(
                "jax_update_excluded_parameter_changed",
                "an excluded parameter changed during the update",
            )
        if entry.logical_path in changed and not moved:
            raise JaxUpdateEvidenceError(
                "jax_update_changed_path_false",
                "a reported changed parameter is byte-identical",
            )
        if entry.logical_path in unchanged and moved:
            raise JaxUpdateEvidenceError(
                "jax_update_unchanged_path_false",
                "a reported unchanged parameter changed",
            )
    if optimizer_before is None or optimizer_after is None:
        return
    for path in set(optimizer_before) & set(optimizer_after):
        state_moved = not bool(
            jnp.array_equal(
                jnp.asarray(optimizer_before[path]),
                jnp.asarray(optimizer_after[path]),
            )
        )
        if path not in selected and state_moved:
            raise JaxUpdateEvidenceError(
                "jax_update_excluded_optimizer_state_changed",
                "an excluded per-parameter optimizer state changed",
            )


def _mapping_leaf(value: Mapping[str, Any], keypath: tuple[str, ...]) -> Any:
    branch: Any = value
    for key in keypath:
        branch = branch[key]
    return branch


def validate_jax_batch_binding(learning_batch: LearningBatch, batch: Any) -> None:
    """Require the JAX values sent to execution to identify their JSON source."""

    from radjax_student.learning.jax_batch import learning_batch_digest
    from radjax_student.learning.jax_core import JaxBatch

    if not isinstance(learning_batch, LearningBatch):
        raise TypeError("learning_batch must be LearningBatch")
    if not isinstance(batch, JaxBatch):
        raise TypeError("batch_materializer must return JaxBatch")
    expected = learning_batch_digest(learning_batch)
    if batch.source_batch_digest != expected:
        raise JaxBatchBindingError(
            "materialized JAX batch source identity does not match LearningBatch"
        )


@dataclass(frozen=True)
class JaxLearningStepExecution:
    """Backend-neutral loop output with JAX values kept outside report models."""

    result: LearningStepResult
    learning_state: LearningState
    optimizer_state: JaxOptimizerState
    parameters: Any
    architecture_carry: Any
    gradients: Any
    runtime_result: ExecutionResult
    objective_descriptor: ObjectiveExecutionDescriptor
    objective_metrics: Mapping[str, Any]
    architecture_metrics: Mapping[str, Any]
    optimizer_metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("objective_metrics", "architecture_metrics", "optimizer_metrics"):
            value = getattr(self, name)
            object.__setattr__(self, name, MappingProxyType(dict(value)))


def execute_jax_learning_step(
    *,
    architecture: JaxArchitecturePlugin,
    objective_selection: ObjectiveRegistrySelection,
    objective_config: ObjectiveConfig,
    objective_descriptor: ObjectiveExecutionDescriptor,
    resolved_objective_selection: ResolvedObjectiveSelection,
    optimizer: JaxOptimizerBackend,
    optimizer_config: OptimizerConfig,
    optimizer_state: JaxOptimizerState,
    learning_state: LearningState,
    architecture_config: ArchitectureConfig,
    parameters: Any,
    architecture_carry: Any,
    learning_batch: LearningBatch,
    batch_materializer: JaxBatchMaterializer,
    parameter_layout: ParameterTreeLayout,
    runtime_key_stream: RuntimeKeyStream | None = None,
    rng_slot: str = "dropout",
    rng_invocation_index: int = 0,
    runtime_context: ExecutionContext,
    runtime_backend: ExecutionBackend,
    execution_request: ExecutionRequest,
    precision_policy: str | None = None,
    schedule_values: Mapping[str, Any] | None = None,
    runtime_callable_binding: RuntimeCallableBinding | None = None,
) -> JaxLearningStepExecution:
    """Run one full JAX update through runtime preparation, dispatch, and receipt."""

    if runtime_callable_binding is not None:
        if not isinstance(runtime_callable_binding, RuntimeCallableBinding):
            raise TypeError("runtime_callable_binding must be RuntimeCallableBinding")
        if execution_request.callable_reference != runtime_callable_binding.reference:
            raise ValueError(
                "execution request does not match runtime callable binding"
            )

    if not isinstance(architecture, JaxArchitecturePlugin):
        raise TypeError("JAX learning requires a complete JaxArchitecturePlugin")
    if not isinstance(objective_selection, ObjectiveRegistrySelection):
        raise TypeError("JAX learning requires ObjectiveRegistrySelection")
    if not objective_selection.is_registry_selected:
        raise ObjectiveContractError(
            "objective_identity_mismatch",
            "JAX learning requires an objective selected through ObjectiveRegistry",
        )
    if not isinstance(objective_config, ObjectiveConfig):
        raise TypeError("JAX learning requires ObjectiveConfig")
    if not isinstance(objective_descriptor, ObjectiveExecutionDescriptor):
        raise TypeError("JAX learning requires ObjectiveExecutionDescriptor")
    if not isinstance(resolved_objective_selection, ResolvedObjectiveSelection):
        raise TypeError("JAX learning requires ResolvedObjectiveSelection")
    if (
        objective_config.identity != objective_selection.identity
        or objective_descriptor.identity != objective_selection.identity
        or objective_descriptor.capability_profile_digest
        != objective_selection.profile.digest
        or objective_descriptor.config_digest != objective_config.digest
        or objective_descriptor.resolved_surface_identity
        != resolved_objective_selection.digest
        or objective_descriptor.metric_schema_id
        != objective_selection.profile.metric_schema_id
        or objective_descriptor.implementation_identity
        != objective_selection.implementation_identity
    ):
        raise ObjectiveContractError(
            "objective_config_identity_mismatch",
            "JAX learning objective identity is inconsistent",
        )
    objective_selection.plugin.validate_config(objective_config)
    objective_selection.plugin.validate_resolved_surface(resolved_objective_selection)
    if not isinstance(optimizer, JaxOptimizerBackend):
        raise TypeError("JAX learning requires a complete JaxOptimizerBackend")
    if not isinstance(optimizer_state, JaxOptimizerState):
        raise TypeError("optimizer_state must be JaxOptimizerState")
    if not isinstance(learning_state, LearningState):
        raise TypeError("learning_state must be LearningState")
    architecture.validate_config(architecture_config)
    validation = architecture.validate_batch(learning_batch, architecture_config)
    if validation.status != "pass":
        raise ValueError("architecture rejected the learning batch")
    if not isinstance(batch_materializer, JaxBatchMaterializer):
        raise TypeError("batch_materializer must implement JaxBatchMaterializer")
    batch = batch_materializer.materialize(learning_batch)
    validate_jax_batch_binding(learning_batch, batch)
    # Validate objective-owned target requirements before compiling the loss.
    # The objective remains surface-only inside the JAX graph.
    objective_selection.plugin.validate_targets(batch.targets)
    stream = (
        runtime_key_stream or RuntimeKeys.from_seed(runtime_context.root_seed).dropout
    )
    expected_stream = RuntimeKeys.from_seed(runtime_context.root_seed).stream(
        stream.name
    )
    if stream != expected_stream:
        raise ValueError("runtime key stream does not belong to the runtime context")
    validate_runtime_jax_key_request(
        context=runtime_context,
        stream=stream,
        global_step=learning_state.global_step,
        micro_step=learning_state.micro_step,
        slot=rng_slot,
        invocation_index=rng_invocation_index,
        expected_coordinates={
            "global_step": learning_state.global_step,
            "micro_step": learning_state.micro_step,
            "invocation_index": rng_invocation_index,
        },
    )
    rng_key = derive_jax_key(
        stream,
        global_step=learning_state.global_step,
        micro_step=learning_state.micro_step,
        slot=rng_slot,
        invocation_index=rng_invocation_index,
    )
    values = {} if schedule_values is None else dict(schedule_values)
    plan = prepare_jax_execution_plan(
        architecture=architecture,
        parameters=parameters,
        parameter_layout=parameter_layout,
        objective_scope=learning_state.active_objective_scope,
        update_scope=learning_state.active_update_scope,
    )
    if plan.objective_selection != resolved_objective_selection:
        raise ObjectiveContractError(
            "objective_surface_identity_mismatch",
            "architecture objective resolution differs from lifecycle selection",
        )
    optimizer_descriptor = optimizer.jax_state_descriptor(parameter_layout)
    validate_jax_optimizer_state(
        optimizer_state,
        optimizer=optimizer,
        optimizer_id=optimizer_config.optimizer_id,
        parameter_layout=parameter_layout,
        descriptor=optimizer_descriptor,
    )
    resolved_precision = precision_policy or architecture_config.dtype_intent
    if resolved_precision == "unspecified":
        resolved_precision = str(
            runtime_context.metadata.get("precision_policy", "automatic")
        )
    if (
        architecture_config.dtype_intent
        not in {
            "unspecified",
            resolved_precision,
        }
        and resolved_precision != "automatic"
    ):
        raise ValueError("runtime precision conflicts with architecture dtype intent")
    prepared_inputs = prepare_jax_inputs(
        backend=runtime_backend,
        context=runtime_context,
        parameters=parameters,
        architecture_carry=architecture_carry,
        optimizer_state=optimizer_state.arrays,
        batch=batch,
        precision_policy=resolved_precision,
    )
    value_and_grad = build_value_and_grad_fn(
        build_registered_jax_loss_fn(
            architecture=architecture,
            objective_selection=objective_selection,
            objective_config=objective_config,
            objective_descriptor=objective_descriptor,
            resolved_selection=resolved_objective_selection,
        )
    )

    def complete_step(
        current_parameters: Any,
        current_carry: Any,
        current_optimizer_arrays: Any,
        current_batch: JaxBatch,
        current_rng_key: Any | None,
        global_step: Any,
        micro_step: Any,
        optimizer_step: Any,
    ):
        (loss_and_auxiliary, gradients) = value_and_grad(
            current_parameters,
            current_carry,
            current_batch,
            current_rng_key,
        )
        loss, auxiliary = loss_and_auxiliary
        (
            updated_parameters,
            updated_optimizer_arrays,
            changed_mask,
            optimizer_metrics,
        ) = optimizer.apply_jax_updates(
            parameters=current_parameters,
            gradients=gradients,
            optimizer_array_state=current_optimizer_arrays,
            update_mask=plan.update_mask,
            config=optimizer_config,
            schedule_values=values,
        )
        return (
            loss,
            auxiliary,
            gradients,
            updated_parameters,
            updated_optimizer_arrays,
            changed_mask,
            optimizer_metrics,
            global_step + 1,
            micro_step * 0,
            optimizer_step + 1,
        )

    step_arguments = (
        prepared_inputs.parameters,
        prepared_inputs.architecture_carry,
        prepared_inputs.optimizer_state,
        prepared_inputs.batch,
        rng_key,
        learning_state.global_step,
        learning_state.micro_step,
        learning_state.optimizer_step,
    )
    execution_kwargs: dict[str, Any] = {
        "context": runtime_context,
        "request": execution_request,
        "backend": runtime_backend,
        "args": step_arguments,
    }
    if runtime_callable_binding is None:
        execution_kwargs["function"] = complete_step
    else:
        execution_kwargs["callable_binding"] = runtime_callable_binding
        # JAX recognizes ``tree_util.Partial`` as a callable pytree.  This
        # keeps the runtime-dispatched target top-level while preserving the
        # owner-built JAX body without passing a raw Python function to JIT.
        from importlib import import_module

        execution_kwargs["args"] = (
            import_module("jax").tree_util.Partial(complete_step),
            *step_arguments,
        )
    output, runtime_result = execute_function(**execution_kwargs)
    if runtime_result.status != "pass" or output is None:
        blocker = runtime_result.blockers[0] if runtime_result.blockers else None
        detail = (
            "" if blocker is None else f": {blocker.message} ({dict(blocker.details)})"
        )
        raise ValueError(
            f"runtime execution failed for complete JAX learning step{detail}"
        )
    runtime_result = replace(
        runtime_result,
        output_metadata={
            **dict(runtime_result.output_metadata),
            "input_preparation": prepared_inputs.metadata,
            "rng_bridge": {
                "schema_version": JAX_KEY_BRIDGE_VERSION,
                "prng_implementation": JAX_PRNG_IMPLEMENTATION,
                "stream": stream.name,
                "slot": rng_slot,
                "global_step": learning_state.global_step,
                "micro_step": learning_state.micro_step,
                "invocation_index": rng_invocation_index,
            },
            "objective_execution_descriptor": objective_descriptor.to_dict(),
        },
    )
    (
        loss,
        auxiliary,
        gradients,
        updated_parameters,
        updated_optimizer_arrays,
        changed_mask,
        optimizer_metrics,
        next_global_step,
        next_micro_step,
        next_optimizer_step,
    ) = output
    if not isinstance(auxiliary, JaxLossAuxiliary):
        raise TypeError("JAX loss must return JaxLossAuxiliary")
    objective_selection.plugin.validate_metrics(auxiliary.objective_metrics)
    validate_finite_loss_and_gradients(loss, gradients)
    require_finite_jax_gradients(optimizer_metrics)
    updated_optimizer_state = advanced_jax_optimizer_state(
        optimizer_state, updated_optimizer_arrays
    )
    validate_jax_optimizer_state(
        updated_optimizer_state,
        optimizer=optimizer,
        optimizer_id=optimizer_config.optimizer_id,
        parameter_layout=parameter_layout,
        descriptor=optimizer_descriptor,
    )
    updated_learning_state = replace(
        learning_state,
        global_step=int(next_global_step),
        micro_step=int(next_micro_step),
        optimizer_step=int(next_optimizer_step),
    )
    changed = _paths_for_mask(parameter_layout, changed_mask, True)
    unchanged = tuple(
        path for path in parameter_layout.logical_paths if path not in set(changed)
    )
    metric_values = {
        **_finite_metric_values(auxiliary.objective_metrics),
        **_finite_metric_values(auxiliary.architecture_metrics),
        **_finite_metric_values(optimizer_metrics),
    }
    metrics = tuple(
        MetricRecord(name, value, updated_learning_state.global_step)
        for name, value in sorted(metric_values.items())
    )
    result = LearningStepResult(
        status="pass",
        global_step_before=learning_state.global_step,
        global_step_after=updated_learning_state.global_step,
        active_update_scope=learning_state.active_update_scope,
        active_objective_scope=learning_state.active_objective_scope,
        loss=LossResult(
            loss=float(loss),
            objective_scope=plan.objective_selection.scope,
            components=_finite_metric_values(auxiliary.objective_metrics),
        ),
        metrics=metrics,
        changed_parameter_paths=changed,
        unchanged_parameter_paths=unchanged,
    )
    return JaxLearningStepExecution(
        result=result,
        learning_state=updated_learning_state,
        optimizer_state=updated_optimizer_state,
        parameters=updated_parameters,
        architecture_carry=auxiliary.updated_architecture_carry,
        gradients=gradients,
        runtime_result=runtime_result,
        objective_descriptor=objective_descriptor,
        objective_metrics=auxiliary.objective_metrics,
        architecture_metrics=auxiliary.architecture_metrics,
        optimizer_metrics=optimizer_metrics,
    )


def _finite_metric_values(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        scalar = float(value)
        if not scalar == scalar or scalar in (float("inf"), float("-inf")):
            raise ValueError("JAX metric values must be finite")
        result[str(name)] = scalar
    return result


def _paths_for_mask(
    layout: ParameterTreeLayout, mask: Any, expected: bool
) -> tuple[str, ...]:
    result = []
    for entry in layout.entries:
        value = mask
        for key in entry.jax_keypath:
            value = value[key]
        if bool(value) is expected:
            result.append(entry.logical_path)
    return tuple(sorted(result))


__all__ = [
    "GENERIC_JAX_LEARNING_STEP_DECLARATION",
    "JaxLearningStepExecution",
    "JaxBatchBindingError",
    "JaxUpdateEvidenceError",
    "execute_jax_learning_step",
    "execute_jax_learning_step_kernel",
    "validate_jax_batch_binding",
    "validate_jax_update_evidence",
]
