"""Complete runtime-hosted JAX learning step for the P3.11 conveyor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import ArchitectureConfig, JaxArchitecturePlugin
from radjax_student.contracts import ParameterTreeLayout
from radjax_student.learning import (
    LearningState,
    LearningStepResult,
    LossResult,
    MetricRecord,
)
from radjax_student.learning.jax_core import (
    JaxBatch,
    JaxLossAuxiliary,
    JaxObjective,
    JaxObjectiveConfig,
    build_resolved_jax_loss_fn,
    build_value_and_grad_fn,
    validate_finite_loss_and_gradients,
)
from radjax_student.learning.jax_execution import prepare_jax_execution_plan
from radjax_student.optimizers import (
    JaxOptimizerExecution,
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
    execute_function,
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
    objective: JaxObjective,
    optimizer: JaxOptimizerExecution,
    optimizer_config: OptimizerConfig,
    optimizer_state: JaxOptimizerState,
    learning_state: LearningState,
    architecture_config: ArchitectureConfig,
    parameters: Any,
    architecture_carry: Any,
    batch: JaxBatch,
    parameter_layout: ParameterTreeLayout,
    rng_key: Any | None,
    runtime_context: ExecutionContext,
    runtime_backend: ExecutionBackend,
    execution_request: ExecutionRequest,
    schedule_values: Mapping[str, Any] | None = None,
) -> JaxLearningStepExecution:
    """Run one full JAX update through runtime preparation, dispatch, and receipt."""

    if not isinstance(architecture, JaxArchitecturePlugin):
        raise TypeError("JAX learning requires a complete JaxArchitecturePlugin")
    if not isinstance(optimizer, JaxOptimizerExecution):
        raise TypeError("JAX learning requires an optimizer JAX capability")
    if not isinstance(optimizer_state, JaxOptimizerState):
        raise TypeError("optimizer_state must be JaxOptimizerState")
    if not isinstance(learning_state, LearningState):
        raise TypeError("learning_state must be LearningState")
    architecture.validate_config(architecture_config)
    if not isinstance(batch, JaxBatch):
        raise TypeError("batch must be JaxBatch")
    values = {} if schedule_values is None else dict(schedule_values)
    plan = prepare_jax_execution_plan(
        architecture=architecture,
        parameters=parameters,
        parameter_layout=parameter_layout,
        objective_scope=learning_state.active_objective_scope,
        update_scope=learning_state.active_update_scope,
    )
    optimizer_descriptor = optimizer.jax_state_descriptor(parameter_layout)
    validate_jax_optimizer_state(
        optimizer_state,
        optimizer_id=optimizer_config.optimizer_id,
        parameter_layout=parameter_layout,
        descriptor=optimizer_descriptor,
    )
    objective_config = JaxObjectiveConfig(
        objective_id="resolved_objective",
        objective_scope=plan.objective_selection.scope,
    )
    value_and_grad = build_value_and_grad_fn(
        build_resolved_jax_loss_fn(architecture, objective, plan.objective_selection)
    )

    def complete_step(
        current_parameters: Any,
        current_carry: Any,
        current_optimizer_arrays: Any,
        current_batch: JaxBatch,
        current_rng_key: Any | None,
    ):
        (loss_and_auxiliary, gradients) = value_and_grad(
            current_parameters,
            current_carry,
            current_batch,
            objective_config,
            current_rng_key,
        )
        loss, auxiliary = loss_and_auxiliary
        updated_parameters, updated_optimizer_arrays, optimizer_metrics = (
            optimizer.apply_jax_updates(
                parameters=current_parameters,
                gradients=gradients,
                optimizer_array_state=current_optimizer_arrays,
                update_mask=plan.update_mask,
                config=optimizer_config,
                schedule_values=values,
            )
        )
        return (
            loss,
            auxiliary,
            gradients,
            updated_parameters,
            updated_optimizer_arrays,
            optimizer_metrics,
        )

    output, runtime_result = execute_function(
        context=runtime_context,
        function=complete_step,
        request=execution_request,
        backend=runtime_backend,
        args=(
            parameters,
            architecture_carry,
            optimizer_state.arrays,
            batch,
            rng_key,
        ),
    )
    if runtime_result.status != "pass" or output is None:
        raise ValueError("runtime execution failed for complete JAX learning step")
    (
        loss,
        auxiliary,
        gradients,
        updated_parameters,
        updated_optimizer_arrays,
        optimizer_metrics,
    ) = output
    if not isinstance(auxiliary, JaxLossAuxiliary):
        raise TypeError("JAX loss must return JaxLossAuxiliary")
    validate_finite_loss_and_gradients(loss, gradients)
    require_finite_jax_gradients(optimizer_metrics)
    updated_optimizer_state = advanced_jax_optimizer_state(
        optimizer_state, updated_optimizer_arrays
    )
    updated_learning_state = replace(
        learning_state,
        global_step=learning_state.global_step + 1,
        micro_step=learning_state.micro_step + 1,
        optimizer_step=learning_state.optimizer_step + 1,
    )
    changed = tuple(sorted(plan.update_selection.selected_parameter_paths))
    unchanged = tuple(sorted(plan.update_selection.excluded_parameter_paths))
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


__all__ = ["JaxLearningStepExecution", "execute_jax_learning_step"]
