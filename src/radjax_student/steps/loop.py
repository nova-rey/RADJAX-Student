"""Bounded generic repetition over the P3.5 single-step seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from radjax_student.architecture import ArchitectureConfig, ArchitecturePlugin
from radjax_student.learning import (
    LearningBatch,
    LearningIssue,
    LearningState,
    MetricRecord,
)
from radjax_student.learning.hooks import (
    HookContext,
    HookPolicy,
    LearningHook,
    dispatch_hooks,
)
from radjax_student.optimizers import OptimizerBackend, OptimizerConfig, OptimizerState
from radjax_student.steps.single import (
    ScalarObjective,
    SingleStepExecution,
    learning_step,
)

DEFAULT_HOOK_POLICY = HookPolicy()

if TYPE_CHECKING:
    from radjax_student.learning.run_report import LearningRunReport


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
    warnings: tuple[LearningIssue, ...] = ()
    hook_events: tuple[str, ...] = ()
    hook_blockers: tuple[LearningIssue, ...] = ()
    report: LearningRunReport | None = None


def _run_learning_loop(
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
    hooks: tuple[LearningHook, ...] = (),
    hook_policy: HookPolicy = DEFAULT_HOOK_POLICY,
) -> LearningLoopResult:
    execution: SingleStepExecution | None = None
    metrics: list[MetricRecord] = []
    checkpoints: list[str] = []
    warnings: list[LearningIssue] = []
    events: list[str] = []
    disabled: tuple[str, ...] = ()
    sequence = 0

    def observe(event: str, current: LearningState, metadata=None):
        nonlocal disabled, sequence
        sequence += 1
        events.append(event)
        dispatch = dispatch_hooks(
            hooks,
            hook_policy,
            HookContext(
                current.run_id,
                event,
                sequence,
                current.global_step,
                tuple(metrics),
                metadata or {},
            ),
            disabled,
        )
        disabled = dispatch.disabled_hook_ids
        metrics.extend(dispatch.metrics)
        warnings.extend(dispatch.warnings)
        return dispatch

    start = observe("loop_start", learning_state)
    if start.blockers:
        return LearningLoopResult(
            "fail",
            execution,
            learning_state.global_step,
            0,
            "hook_failure",
            tuple(metrics[-config.metric_history_limit :]),
            (),
            tuple(warnings),
            tuple(events),
        )
    for _ in range(config.max_steps):
        batch = batch_source.next_batch()
        if batch is None:
            terminal = observe("loop_end", learning_state)
            return LearningLoopResult(
                "fail" if terminal.blockers else "pass",
                execution,
                learning_state.global_step,
                learning_state.global_step,
                "hook_failure" if terminal.blockers else "source_exhausted",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
            )
        dispatched = observe("batch_received", learning_state)
        if dispatched.blockers:
            return LearningLoopResult(
                "fail",
                execution,
                learning_state.global_step,
                learning_state.global_step,
                "hook_failure",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
            )
        dispatched = observe("step_start", learning_state)
        if dispatched.blockers:
            return LearningLoopResult(
                "fail",
                execution,
                learning_state.global_step,
                learning_state.global_step,
                "hook_failure",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
            )
        try:
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
        except Exception as exc:
            failure_dispatch = observe(
                "failure",
                learning_state,
                {
                    "failure_stage": "learning_step",
                    "exception_type": type(exc).__name__,
                },
            )
            return LearningLoopResult(
                "fail",
                execution,
                learning_state.global_step,
                learning_state.global_step,
                "learning_step_failure",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
                failure_dispatch.blockers,
            )
        learning_state, optimizer_state, parameters = (
            execution.learning_state,
            execution.optimizer_state,
            execution.parameters,
        )
        metrics.extend(execution.result.metrics)
        dispatched = observe("step_end", learning_state)
        if dispatched.blockers:
            return LearningLoopResult(
                "fail",
                execution,
                learning_state.global_step,
                learning_state.global_step,
                "hook_failure",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
            )
        if (
            checkpoint
            and config.checkpoint_every_n_steps
            and learning_state.global_step % config.checkpoint_every_n_steps == 0
        ):
            try:
                checkpoints.append(checkpoint(execution))
            except Exception as exc:
                failure_dispatch = observe(
                    "failure",
                    learning_state,
                    {
                        "failure_stage": "checkpoint",
                        "exception_type": type(exc).__name__,
                    },
                )
                return LearningLoopResult(
                    "fail",
                    execution,
                    learning_state.global_step,
                    learning_state.global_step,
                    "checkpoint_failure",
                    tuple(metrics[-config.metric_history_limit :]),
                    tuple(checkpoints),
                    tuple(warnings),
                    tuple(events),
                    failure_dispatch.blockers,
                )
            checkpoint_dispatch = observe("checkpoint", learning_state)
            if checkpoint_dispatch.blockers:
                return LearningLoopResult(
                    "fail",
                    execution,
                    learning_state.global_step,
                    learning_state.global_step,
                    "hook_failure",
                    tuple(metrics[-config.metric_history_limit :]),
                    tuple(checkpoints),
                    tuple(warnings),
                    tuple(events),
                )
    terminal = observe("loop_end", learning_state)
    return LearningLoopResult(
        "fail" if terminal.blockers else "pass",
        execution,
        learning_state.global_step,
        learning_state.global_step,
        "hook_failure" if terminal.blockers else "max_steps",
        tuple(metrics[-config.metric_history_limit :]),
        tuple(checkpoints),
        tuple(warnings),
        tuple(events),
    )


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
    hooks: tuple[LearningHook, ...] = (),
    hook_policy: HookPolicy = DEFAULT_HOOK_POLICY,
    emit_run_report: bool = False,
) -> LearningLoopResult:
    """Run generic learning and optionally attach a post-completion report."""

    result = _run_learning_loop(
        config=config,
        architecture=architecture,
        architecture_config=architecture_config,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        learning_state=learning_state,
        parameters=parameters,
        objective=objective,
        batch_source=batch_source,
        checkpoint=checkpoint,
        hooks=hooks,
        hook_policy=hook_policy,
    )
    if not emit_run_report:
        return result
    from dataclasses import replace

    from radjax_student.learning.run_report import build_learning_run_report

    report_state = (
        result.final_execution.learning_state
        if result.final_execution is not None
        else learning_state
    )
    try:
        report = build_learning_run_report(
            loop_result=result,
            run_id=report_state.run_id,
            update_scope=report_state.active_update_scope.kind,
            objective_scope=report_state.active_objective_scope.kind,
        )
    except (TypeError, ValueError):
        return result
    return replace(result, report=report)


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
