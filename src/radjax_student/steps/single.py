"""The narrow P3.5 composition of generic architecture and optimizer contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from radjax_student.architecture import (
    ArchitectureConfig,
    ArchitecturePlugin,
    ForwardRequest,
)
from radjax_student.learning import (
    LearningBatch,
    LearningState,
    LearningStepResult,
    LossResult,
    MetricRecord,
)
from radjax_student.optimizers import (
    GradientTree,
    OptimizerBackend,
    OptimizerConfig,
    OptimizerState,
    OptimizerUpdateRequest,
)


class ScalarObjective(Protocol):
    """A test objective that exposes a finite loss and stable scalar gradients."""

    def evaluate(
        self, parameters: Mapping[str, float], batch: LearningBatch
    ) -> tuple[float, Mapping[str, float]]: ...


@dataclass(frozen=True)
class SingleStepExecution:
    result: LearningStepResult
    learning_state: LearningState
    optimizer_state: OptimizerState
    parameters: Mapping[str, float]


def learning_step(
    *,
    batch: LearningBatch,
    architecture: ArchitecturePlugin,
    architecture_config: ArchitectureConfig,
    optimizer: OptimizerBackend,
    optimizer_config: OptimizerConfig,
    optimizer_state: OptimizerState,
    learning_state: LearningState,
    parameters: Mapping[str, float],
    objective: ScalarObjective,
) -> SingleStepExecution:
    """Execute exactly one deterministic scalar-contract learning step."""
    validation = architecture.validate_batch(batch, architecture_config)
    if not validation.ok:
        raise ValueError("learning_batch_invalid: architecture rejected batch")
    metadata = architecture.architecture_metadata()
    architecture.resolve_objective_scope(batch.objective_scope, metadata)
    architecture.forward(
        ForwardRequest(batch=batch, parameters=parameters, training=True)
    )
    loss_value, gradient_values = objective.evaluate(parameters, batch)
    if not math.isfinite(loss_value):
        raise ValueError("learning_objective_failed: loss must be finite")
    catalog = architecture.describe_parameters()
    selection = architecture.resolve_update_scope(
        learning_state.active_update_scope, catalog
    )
    gradients = GradientTree(
        catalog.paths, values=gradient_values, metadata={"source": "scalar_objective"}
    )
    update = optimizer.apply_updates(
        OptimizerUpdateRequest(
            gradients,
            optimizer_state,
            optimizer_config,
            selection,
            learning_state.global_step,
            parameters=parameters,
        )
    )
    metrics = (
        MetricRecord("loss", loss_value, learning_state.global_step + 1),
        MetricRecord(
            "gradient_norm",
            math.sqrt(sum(value * value for value in gradient_values.values())),
            learning_state.global_step + 1,
        ),
        MetricRecord(
            "parameter_norm",
            math.sqrt(
                sum(value * value for value in update.updated_parameters.values())
            ),
            learning_state.global_step + 1,
        ),
        MetricRecord(
            "learning_rate",
            update.update_metadata["learning_rate"],
            learning_state.global_step + 1,
        ),
        MetricRecord(
            "changed_parameter_count",
            len(update.changed_parameter_paths),
            learning_state.global_step + 1,
        ),
        MetricRecord(
            "unchanged_parameter_count",
            len(update.unchanged_parameter_paths),
            learning_state.global_step + 1,
        ),
        MetricRecord("step_time", 0.0, learning_state.global_step + 1, unit="seconds"),
    )
    loss = LossResult(
        loss=loss_value, objective_scope=batch.objective_scope, metrics=(metrics[0],)
    )
    result = LearningStepResult(
        status="pass",
        global_step_before=learning_state.global_step,
        global_step_after=learning_state.global_step + 1,
        active_update_scope=learning_state.active_update_scope,
        active_objective_scope=batch.objective_scope,
        loss=loss,
        metrics=metrics,
        changed_parameter_paths=update.changed_parameter_paths,
        unchanged_parameter_paths=update.unchanged_parameter_paths,
    )
    state = LearningState(
        run_id=learning_state.run_id,
        global_step=learning_state.global_step + 1,
        micro_step=0,
        epoch=learning_state.epoch,
        optimizer_step=learning_state.optimizer_step + 1,
        runtime_state_reference=learning_state.runtime_state_reference,
        active_update_scope=learning_state.active_update_scope,
        active_objective_scope=batch.objective_scope,
        metadata=learning_state.metadata,
    )
    return SingleStepExecution(
        result, state, update.updated_optimizer_state, update.updated_parameters
    )
