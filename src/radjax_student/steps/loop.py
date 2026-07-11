"""Bounded generic repetition over the P3.5 single-step seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from radjax_student.architecture import ArchitectureConfig, ArchitecturePlugin
from radjax_student.learning import LearningBatch, LearningState, MetricRecord
from radjax_student.optimizers import OptimizerBackend, OptimizerConfig, OptimizerState
from radjax_student.steps.single import (
    ScalarObjective,
    SingleStepExecution,
    learning_step,
)


class BatchSource(Protocol):
    source_id: str

    def next_batch(self) -> LearningBatch | None: ...
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


@dataclass(frozen=True)
class LearningLoopConfig:
    max_steps: int
    gradient_accumulation_steps: int = 1
    metric_history_limit: int = 64
    checkpoint_every_n_steps: int | None = None
    fail_fast: bool = True

    def __post_init__(self) -> None:
        if (
            self.max_steps < 0
            or self.gradient_accumulation_steps != 1
            or self.metric_history_limit < 1
        ):
            raise ValueError("loop config is invalid or accumulation is unsupported")
        if (
            self.checkpoint_every_n_steps is not None
            and self.checkpoint_every_n_steps < 1
        ):
            raise ValueError("checkpoint interval must be positive")


@dataclass(frozen=True)
class LearningLoopResult:
    status: str
    final_execution: SingleStepExecution | None
    steps_completed: int
    batches_consumed: int
    stop_reason: str
    metrics: tuple[MetricRecord, ...]
    checkpoints: tuple[str, ...] = ()


def run_learning_loop(
    *,
    config: LearningLoopConfig,
    architecture: ArchitecturePlugin,
    architecture_config: ArchitectureConfig,
    optimizer: OptimizerBackend,
    optimizer_config: OptimizerConfig,
    optimizer_state: OptimizerState,
    learning_state: LearningState,
    parameters: Mapping[str, float],
    objective: ScalarObjective,
    batch_source: BatchSource,
    checkpoint: Callable[[SingleStepExecution], str] | None = None,
) -> LearningLoopResult:
    execution: SingleStepExecution | None = None
    metrics: list[MetricRecord] = []
    checkpoints: list[str] = []
    for _ in range(config.max_steps):
        batch = batch_source.next_batch()
        if batch is None:
            return LearningLoopResult(
                "pass",
                execution,
                learning_state.global_step,
                learning_state.global_step,
                "source_exhausted",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
            )
        execution = learning_step(
            batch=batch,
            architecture=architecture,
            architecture_config=architecture_config,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            optimizer_state=optimizer_state,
            learning_state=learning_state,
            parameters=parameters,
            objective=objective,
        )
        learning_state, optimizer_state, parameters = (
            execution.learning_state,
            execution.optimizer_state,
            execution.parameters,
        )
        metrics.extend(execution.result.metrics)
        if (
            checkpoint
            and config.checkpoint_every_n_steps
            and learning_state.global_step % config.checkpoint_every_n_steps == 0
        ):
            checkpoints.append(checkpoint(execution))
    return LearningLoopResult(
        "pass",
        execution,
        learning_state.global_step,
        learning_state.global_step,
        "max_steps",
        tuple(metrics[-config.metric_history_limit :]),
        tuple(checkpoints),
    )


@dataclass
class SyntheticBatchSource:
    batches: tuple[LearningBatch, ...]
    source_id: str = "synthetic.v1"
    position: int = 0

    def next_batch(self) -> LearningBatch | None:
        if self.position >= len(self.batches):
            return None
        batch = self.batches[self.position]
        self.position += 1
        return batch

    def state_dict(self) -> Mapping[str, object]:
        return {
            "source_id": self.source_id,
            "position": self.position,
            "exhausted": self.position >= len(self.batches),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("source_id") != self.source_id:
            raise ValueError("batch source state mismatch")
        self.position = int(state["position"])
