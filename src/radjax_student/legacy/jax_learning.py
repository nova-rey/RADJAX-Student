"""Explicit pre-P3.11 JAX update adapter retained for compatibility evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import JaxArchitectureExecution
from radjax_student.learning.jax_core import (
    JaxBatch,
    JaxLossAuxiliary,
    JaxObjective,
    JaxObjectiveConfig,
    apply_scoped_gradient_update,
    build_jax_loss_fn,
    build_value_and_grad_fn,
    validate_finite_loss_and_gradients,
)
from radjax_student.runtime import (
    ExecutionBackend,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult,
    execute_function,
)


@dataclass(frozen=True)
class LegacyJaxLearningStepExecution:
    loss: Any
    parameters: Any
    architecture_carry: Any
    gradients: Any
    objective_metrics: Mapping[str, Any]
    architecture_metrics: Mapping[str, Any]
    runtime_result: ExecutionResult

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "objective_metrics", MappingProxyType(dict(self.objective_metrics))
        )
        object.__setattr__(
            self,
            "architecture_metrics",
            MappingProxyType(dict(self.architecture_metrics)),
        )


def execute_legacy_jax_learning_step(
    *,
    architecture: JaxArchitectureExecution,
    objective: JaxObjective,
    parameters: Any,
    architecture_carry: Any,
    batch: JaxBatch,
    objective_config: JaxObjectiveConfig,
    rng_key: Any | None,
    selection_mask: Any,
    learning_rate: float,
    runtime_context: ExecutionContext,
    runtime_backend: ExecutionBackend,
    execution_request: ExecutionRequest,
) -> LegacyJaxLearningStepExecution:
    """Compatibility-only partial update path; production uses `steps.jax_step`."""

    loss_and_grad = build_value_and_grad_fn(build_jax_loss_fn(architecture, objective))
    output, runtime_result = execute_function(
        context=runtime_context,
        function=loss_and_grad,
        request=execution_request,
        backend=runtime_backend,
        args=(parameters, architecture_carry, batch, objective_config, rng_key),
    )
    if runtime_result.status != "pass" or output is None:
        raise ValueError("runtime execution failed for legacy JAX learning step")
    (loss, auxiliary), gradients = output
    if not isinstance(auxiliary, JaxLossAuxiliary):
        raise TypeError("JAX loss must return JaxLossAuxiliary")
    validate_finite_loss_and_gradients(loss, gradients)
    return LegacyJaxLearningStepExecution(
        loss=loss,
        parameters=apply_scoped_gradient_update(
            parameters, gradients, selection_mask, learning_rate
        ),
        architecture_carry=auxiliary.updated_architecture_carry,
        gradients=gradients,
        objective_metrics=auxiliary.objective_metrics,
        architecture_metrics=auxiliary.architecture_metrics,
        runtime_result=runtime_result,
    )


__all__ = ["LegacyJaxLearningStepExecution", "execute_legacy_jax_learning_step"]
