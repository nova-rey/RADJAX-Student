"""Explicit legacy scalar handcrafted-gradient compatibility path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

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


class LegacyScalarObjective(Protocol):
    """Historical objective protocol that receives scalar parameters directly."""

    def evaluate(
        self, parameters: Mapping[str, float], batch: LearningBatch
    ) -> tuple[float, Mapping[str, float]]: ...


@dataclass(frozen=True)
class LegacyScalarObjectiveAdapter:
    objective: LegacyScalarObjective

    def evaluate(
        self, parameters: Mapping[str, float], batch: LearningBatch
    ) -> tuple[float, Mapping[str, float]]:
        return self.objective.evaluate(parameters, batch)


@dataclass(frozen=True)
class LegacyScalarStepExecution:
    result: LearningStepResult
    learning_state: LearningState
    optimizer_state: OptimizerState
    parameters: Mapping[str, float]


def legacy_scalar_learning_step(
    *,
    batch: LearningBatch,
    architecture: ArchitecturePlugin,
    architecture_config: ArchitectureConfig,
    optimizer: OptimizerBackend,
    optimizer_config: OptimizerConfig,
    optimizer_state: OptimizerState,
    learning_state: LearningState,
    parameters: Mapping[str, float],
    objective: LegacyScalarObjective,
) -> LegacyScalarStepExecution:
    """Run the quarantined Phase 3 scalar compatibility seam."""

    validation = architecture.validate_batch(batch, architecture_config)
    if not validation.ok:
        raise ValueError("learning_batch_invalid: architecture rejected batch")
    metadata = architecture.architecture_metadata()
    architecture.resolve_objective_scope(batch.objective_scope, metadata)
    architecture.forward(
        ForwardRequest(batch=batch, parameters=parameters, training=True)
    )
    loss_value, gradient_values = LegacyScalarObjectiveAdapter(objective).evaluate(
        parameters, batch
    )
    if not math.isfinite(loss_value):
        raise ValueError("learning_objective_failed: loss must be finite")
    catalog = architecture.describe_parameters()
    selection = architecture.resolve_update_scope(
        learning_state.active_update_scope, catalog
    )
    gradients = GradientTree(
        catalog.paths, values=gradient_values, metadata={"source": "legacy_scalar"}
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
    step = learning_state.global_step + 1
    metrics = (
        MetricRecord("loss", loss_value, step),
        MetricRecord(
            "gradient_norm",
            math.sqrt(sum(value * value for value in gradient_values.values())),
            step,
        ),
        MetricRecord(
            "parameter_norm",
            math.sqrt(
                sum(value * value for value in update.updated_parameters.values())
            ),
            step,
        ),
        MetricRecord("learning_rate", update.update_metadata["learning_rate"], step),
        MetricRecord(
            "changed_parameter_count", len(update.changed_parameter_paths), step
        ),
        MetricRecord(
            "unchanged_parameter_count", len(update.unchanged_parameter_paths), step
        ),
        MetricRecord("step_time", 0.0, step, unit="seconds"),
    )
    result = LearningStepResult(
        status="pass",
        global_step_before=learning_state.global_step,
        global_step_after=step,
        active_update_scope=learning_state.active_update_scope,
        active_objective_scope=batch.objective_scope,
        loss=LossResult(
            loss=loss_value,
            objective_scope=batch.objective_scope,
            metrics=(metrics[0],),
        ),
        metrics=metrics,
        changed_parameter_paths=update.changed_parameter_paths,
        unchanged_parameter_paths=update.unchanged_parameter_paths,
    )
    state = LearningState(
        run_id=learning_state.run_id,
        global_step=step,
        micro_step=0,
        epoch=learning_state.epoch,
        optimizer_step=learning_state.optimizer_step + 1,
        runtime_state_reference=learning_state.runtime_state_reference,
        active_update_scope=learning_state.active_update_scope,
        active_objective_scope=batch.objective_scope,
        metadata=learning_state.metadata,
    )
    return LegacyScalarStepExecution(
        result, state, update.updated_optimizer_state, update.updated_parameters
    )


def run_legacy_learning_loop(**kwargs: Any):
    """Run the generic loop with the explicit legacy scalar executor."""

    from radjax_student.steps.loop import run_learning_loop

    return run_learning_loop(step_executor=legacy_scalar_learning_step, **kwargs)


__all__ = [
    "LegacyScalarObjective",
    "LegacyScalarObjectiveAdapter",
    "LegacyScalarStepExecution",
    "legacy_scalar_learning_step",
    "run_legacy_learning_loop",
]
