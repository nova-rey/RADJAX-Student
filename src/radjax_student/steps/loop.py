"""Bounded generic repetition over the P3.5 single-step seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from radjax_student.architecture import ArchitectureConfig, ArchitecturePlugin
from radjax_student.learning import (
    LearningBatch,
    LearningIssue,
    LearningState,
    LearningStepResult,
    MetricRecord,
)
from radjax_student.learning.hooks import (
    HookContext,
    HookPolicy,
    LearningHook,
    dispatch_hooks,
)
from radjax_student.optimizers import OptimizerBackend, OptimizerConfig, OptimizerState

DEFAULT_HOOK_POLICY = HookPolicy()


class LearningLoopBoundaryError(ValueError):
    """Stable failure when a completed generic-loop result is not successful."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def validate_learning_step_execution(
    execution: LearningStepExecutionProtocol,
    *,
    previous_learning_state: LearningState,
    previous_optimizer_state: Any,
) -> None:
    """Validate the backend-neutral transition before generic loop adoption.

    The loop deliberately does not interpret optimizer arrays or model carry,
    but it must reject an executor that claims a completed update without the
    corresponding immutable learning and optimizer envelope transition.
    """

    if not isinstance(getattr(execution, "result", None), LearningStepResult):
        raise LearningLoopBoundaryError(
            "learning_step_result_invalid", "step executor did not return an execution"
        )
    if not isinstance(getattr(execution, "learning_state", None), LearningState):
        raise LearningLoopBoundaryError(
            "learning_state_invalid", "step execution lacks LearningState"
        )
    state = execution.learning_state
    if state.global_step != previous_learning_state.global_step + 1:
        raise LearningLoopBoundaryError(
            "learning_global_step_invalid", "completed step did not advance global step"
        )
    if state.optimizer_step != previous_learning_state.optimizer_step + 1:
        raise LearningLoopBoundaryError(
            "learning_optimizer_step_invalid",
            "completed step did not advance optimizer step",
        )
    if state.micro_step != 0:
        raise LearningLoopBoundaryError(
            "learning_micro_step_invalid",
            "micro step must reset when accumulation is unsupported",
        )
    optimizer = getattr(execution, "optimizer_state", None)
    envelope = getattr(optimizer, "envelope", optimizer)
    previous_envelope = getattr(
        previous_optimizer_state, "envelope", previous_optimizer_state
    )
    if hasattr(envelope, "step") and hasattr(previous_envelope, "step"):
        if envelope.step != previous_envelope.step + 1:
            raise LearningLoopBoundaryError(
                "learning_optimizer_envelope_invalid",
                "completed step did not advance optimizer envelope",
            )


if TYPE_CHECKING:
    from radjax_student.learning.run_report import LearningRunReport


class BatchSource(Protocol):
    source_id: str

    def next_batch(self) -> LearningBatch | None: ...
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class LearningStepExecutionProtocol(Protocol):
    result: Any
    learning_state: LearningState
    optimizer_state: Any
    parameters: Any


class LearningStepExecutor(Protocol):
    def __call__(self, **kwargs: Any) -> LearningStepExecutionProtocol: ...


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
    final_execution: LearningStepExecutionProtocol | None
    steps_completed: int
    global_step: int
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
    parameters: Any,
    objective: Any,
    batch_source: BatchSource,
    step_executor: LearningStepExecutor,
    checkpoint: Callable[[LearningStepExecutionProtocol], str] | None = None,
    hooks: tuple[LearningHook, ...] = (),
    hook_policy: HookPolicy = DEFAULT_HOOK_POLICY,
) -> LearningLoopResult:
    execution: LearningStepExecutionProtocol | None = None
    steps_completed = 0
    batches_consumed = 0
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
            steps_completed,
            learning_state.global_step,
            0,
            "hook_failure",
            tuple(metrics[-config.metric_history_limit :]),
            (),
            tuple(warnings),
            tuple(events),
            start.blockers,
        )
    for _ in range(config.max_steps):
        batch = batch_source.next_batch()
        if batch is None:
            terminal = observe("loop_end", learning_state)
            return LearningLoopResult(
                "fail" if terminal.blockers else "pass",
                execution,
                steps_completed,
                learning_state.global_step,
                batches_consumed,
                "hook_failure" if terminal.blockers else "source_exhausted",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
                terminal.blockers,
            )
        batches_consumed += 1
        dispatched = observe("batch_received", learning_state)
        if dispatched.blockers:
            return LearningLoopResult(
                "fail",
                execution,
                steps_completed,
                learning_state.global_step,
                batches_consumed,
                "hook_failure",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
                dispatched.blockers,
            )
        dispatched = observe("step_start", learning_state)
        if dispatched.blockers:
            return LearningLoopResult(
                "fail",
                execution,
                steps_completed,
                learning_state.global_step,
                batches_consumed,
                "hook_failure",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
                dispatched.blockers,
            )
        try:
            execution = step_executor(
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
            validate_learning_step_execution(
                execution,
                previous_learning_state=learning_state,
                previous_optimizer_state=optimizer_state,
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
                steps_completed,
                learning_state.global_step,
                batches_consumed,
                str(getattr(exc, "code", "learning_step_failure")),
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
        steps_completed += 1
        metrics.extend(execution.result.metrics)
        dispatched = observe("step_end", learning_state)
        if dispatched.blockers:
            return LearningLoopResult(
                "fail",
                execution,
                steps_completed,
                learning_state.global_step,
                batches_consumed,
                "hook_failure",
                tuple(metrics[-config.metric_history_limit :]),
                tuple(checkpoints),
                tuple(warnings),
                tuple(events),
                dispatched.blockers,
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
                    steps_completed,
                    learning_state.global_step,
                    batches_consumed,
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
                    steps_completed,
                    learning_state.global_step,
                    batches_consumed,
                    "hook_failure",
                    tuple(metrics[-config.metric_history_limit :]),
                    tuple(checkpoints),
                    tuple(warnings),
                    tuple(events),
                    checkpoint_dispatch.blockers,
                )
    terminal = observe("loop_end", learning_state)
    return LearningLoopResult(
        "fail" if terminal.blockers else "pass",
        execution,
        steps_completed,
        learning_state.global_step,
        batches_consumed,
        "hook_failure" if terminal.blockers else "max_steps",
        tuple(metrics[-config.metric_history_limit :]),
        tuple(checkpoints),
        tuple(warnings),
        tuple(events),
        terminal.blockers,
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
    parameters: Any,
    objective: Any,
    batch_source: BatchSource,
    step_executor: LearningStepExecutor,
    checkpoint: Callable[[LearningStepExecutionProtocol], str] | None = None,
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
        step_executor=step_executor,
        checkpoint=checkpoint,
        hooks=hooks,
        hook_policy=hook_policy,
    )
    if not emit_run_report:
        return result
    from dataclasses import replace

    from radjax_student.learning.run_report import build_learning_run_report

    candidate_report_state = (
        getattr(result.final_execution, "learning_state", None)
        if result.final_execution is not None
        else None
    )
    report_state = (
        candidate_report_state
        if isinstance(candidate_report_state, LearningState)
        else learning_state
    )
    try:
        objective_descriptor = (
            getattr(result.final_execution, "objective_descriptor", None)
            if result.final_execution is not None
            else None
        )
        report = build_learning_run_report(
            loop_result=result,
            run_id=report_state.run_id,
            update_scope=report_state.active_update_scope.kind,
            objective_scope=report_state.active_objective_scope.kind,
            objective_descriptor=objective_descriptor,
        )
    except (TypeError, ValueError):
        return result
    return replace(result, report=report)


def require_successful_learning_loop(result: LearningLoopResult) -> LearningLoopResult:
    """Turn a real terminal loop result into a public failure when required."""

    if not isinstance(result, LearningLoopResult):
        raise TypeError("learning loop result is required")
    if result.status != "pass" or result.stop_reason != "max_steps":
        raise LearningLoopBoundaryError(
            f"learning_loop_{result.stop_reason}",
            "generic learning loop did not complete successfully",
        )
    return result


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
