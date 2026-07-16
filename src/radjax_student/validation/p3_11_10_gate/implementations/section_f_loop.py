"""Literal Section F generic-loop, hook, metric, and report experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from radjax_student.architecture import ArchitectureConfig
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.learning import (
    HookPolicy,
    HookResult,
    LearningBatch,
    LearningState,
    LearningStepResult,
    MetricRecord,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerState, SgdOptimizer
from radjax_student.steps.loop import (
    LearningLoopConfig,
    SyntheticBatchSource,
    require_successful_learning_loop,
    run_learning_loop,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_memory_experiment,
    public_boundary,
)


@dataclass
class _Execution:
    result: Any
    learning_state: LearningState
    optimizer_state: OptimizerState
    parameters: Any


class _AdvancingExecutor:
    def __call__(self, **kwargs: Any) -> _Execution:
        state = kwargs["learning_state"]
        optimizer_state = kwargs["optimizer_state"]
        next_state = replace(
            state,
            global_step=state.global_step + 1,
            micro_step=0,
            optimizer_step=state.optimizer_step + 1,
        )
        next_optimizer = replace(optimizer_state, step=optimizer_state.step + 1)
        result = LearningStepResult(
            "pass",
            state.global_step,
            next_state.global_step,
            metrics=(MetricRecord("loss", 1.0, next_state.global_step),),
        )
        return _Execution(result, next_state, next_optimizer, kwargs["parameters"])


class _RaisingExecutor:
    def __call__(self, **kwargs: Any) -> _Execution:
        del kwargs
        raise RuntimeError("literal step executor failure")


class _UnchangedExecutor:
    def __call__(self, **kwargs: Any) -> _Execution:
        state = kwargs["learning_state"]
        return _Execution(
            LearningStepResult("pass", state.global_step, state.global_step),
            state,
            kwargs["optimizer_state"],
            kwargs["parameters"],
        )


class _MalformedResultExecutor:
    def __call__(self, **kwargs: Any) -> object:
        del kwargs
        return object()


class _LegacyPartialExecutor:
    def __call__(self, **kwargs: Any) -> object:
        del kwargs
        from radjax_student.legacy.scalar_learning import LegacyScalarStepExecution

        return LegacyScalarStepExecution(
            LearningStepResult("pass", 0, 1),
            None,
            None,
            {},
        )


class _EventHook:
    hook_id = "literal.event"
    priority = 0
    supported_events = (
        "loop_start",
        "batch_received",
        "step_start",
        "step_end",
        "checkpoint",
        "loop_end",
        "failure",
    )

    def __init__(self, failing_event: str | None = None) -> None:
        self.failing_event = failing_event

    def on_event(self, context: Any) -> HookResult:
        if context.event_type == self.failing_event:
            return HookResult("fail", metadata={"event": context.event_type})
        return HookResult()


@dataclass
class _LoopInput:
    config: LearningLoopConfig
    source: SyntheticBatchSource
    executor: Any
    hooks: tuple[Any, ...] = ()
    checkpoint: Any = None


def _batch(label: str = "batch") -> LearningBatch:
    return LearningBatch(label, inputs={"x": 1.0}, targets={"y": 2.0})


def _input(
    *,
    max_steps: int = 1,
    source: SyntheticBatchSource | None = None,
    executor: Any | None = None,
    hooks: tuple[Any, ...] = (),
    checkpoint: Any = None,
) -> _LoopInput:
    return _LoopInput(
        LearningLoopConfig(
            max_steps=max_steps, checkpoint_every_n_steps=1 if checkpoint else None
        ),
        source or SyntheticBatchSource((_batch(),)),
        executor or _AdvancingExecutor(),
        hooks,
        checkpoint,
    )


@public_boundary("loop_executor_validation")
def _run(value: _LoopInput) -> Any:
    architecture = FakeArchitecturePlugin()
    optimizer = SgdOptimizer()
    result = run_learning_loop(
        config=value.config,
        architecture=architecture,
        architecture_config=ArchitectureConfig(architecture.architecture_id),
        optimizer=optimizer,
        optimizer_config=OptimizerConfig(optimizer.optimizer_id),
        optimizer_state=OptimizerState(
            optimizer.optimizer_id, architecture.describe_parameters().paths
        ),
        learning_state=LearningState("literal-loop"),
        parameters={},
        objective=object(),
        batch_source=value.source,
        step_executor=value.executor,
        checkpoint=value.checkpoint,
        hooks=value.hooks,
        hook_policy=HookPolicy("fail_fast"),
        emit_run_report=True,
    )
    return require_successful_learning_loop(result)


def _record(
    context: GateExecutionContext,
    baseline: Any,
    mutated: Any,
    path: str,
    operation: str,
    public_callable: Any = _run,
    baseline_callable: Any | None = _run,
) -> ExperimentExecution:
    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="generic_learning_loop_input",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=public_callable,
        baseline_callable=baseline_callable,
    )


def experiment_f_normal_loop_deterministic_hooks_metrics_report(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(max_steps=1),
        _input(
            max_steps=2, source=SyntheticBatchSource((_batch("one"), _batch("two")))
        ),
        "config.max_steps",
        "increase_valid_loop_length",
    )


def experiment_f_uninterrupted_resumed_reports_match_within_mode(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(max_steps=1),
        _input(max_steps=1, source=SyntheticBatchSource((_batch("resumed"),))),
        "batch_source.batches[0].batch_id",
        "replace_valid_resumed_batch_identity",
    )


def experiment_f_unsupported_gradient_accumulation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"max_steps": 1, "gradient_accumulation_steps": 1}
    mutated = {"max_steps": 1, "gradient_accumulation_steps": 2}

    @public_boundary("loop_executor_validation")
    def config(value: dict[str, int]) -> Any:
        return LearningLoopConfig(**value)

    return _record(
        context,
        baseline,
        mutated,
        "gradient_accumulation_steps",
        "set_unsupported_gradient_accumulation",
        config,
        config,
    )


def experiment_f_batch_exhaustion_before_required_steps(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(max_steps=1),
        _input(max_steps=2, source=SyntheticBatchSource((_batch(),))),
        "batch_source.length",
        "shorten_batch_source_before_required_steps",
    )


def experiment_f_hook_blocker_loop_start(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(hooks=(_EventHook("loop_start"),)),
        "hooks[0].failing_event",
        "fail_hook_at_loop_start",
    )


def experiment_f_hook_blocker_before_step(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(hooks=(_EventHook("step_start"),)),
        "hooks[0].failing_event",
        "fail_hook_before_step",
    )


def experiment_f_hook_blocker_after_step(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(hooks=(_EventHook("step_end"),)),
        "hooks[0].failing_event",
        "fail_hook_after_step",
    )


def experiment_f_hook_failure_checkpoint_event(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(
            hooks=(_EventHook("checkpoint"),), checkpoint=lambda execution: "checkpoint"
        ),
        "hooks[0].failing_event",
        "fail_hook_at_checkpoint_event",
    )


def experiment_f_step_executor_exception(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(executor=_RaisingExecutor()),
        "step_executor",
        "replace_advancing_executor_with_raising_executor",
    )


def experiment_f_checkpoint_callback_exception(
    context: GateExecutionContext,
) -> ExperimentExecution:
    def failing_checkpoint(execution: Any) -> str:
        del execution
        raise RuntimeError("literal checkpoint callback failure")

    return _record(
        context,
        _input(),
        _input(checkpoint=failing_checkpoint),
        "checkpoint",
        "replace_checkpoint_callback_with_raising_callback",
    )


def experiment_f_malformed_step_execution_result(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(executor=_MalformedResultExecutor()),
        "step_executor.result",
        "replace_execution_result_with_untyped_object",
    )


def experiment_f_legacy_partial_step_execution_result(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(executor=_LegacyPartialExecutor()),
        "step_executor",
        "supply_legacy_partial_step_executor",
    )


def experiment_f_learning_state_fails_advance(
    context: GateExecutionContext,
) -> ExperimentExecution:
    return _record(
        context,
        _input(),
        _input(executor=_UnchangedExecutor()),
        "execution.learning_state.global_step",
        "retain_learning_global_step_after_execution",
    )


def experiment_f_global_step_without_optimizer_step(
    context: GateExecutionContext,
) -> ExperimentExecution:
    class GlobalOnlyExecutor(_AdvancingExecutor):
        def __call__(self, **kwargs: Any) -> _Execution:
            result = super().__call__(**kwargs)
            result.learning_state = replace(result.learning_state, optimizer_step=0)
            return result

    return _record(
        context,
        _input(),
        _input(executor=GlobalOnlyExecutor()),
        "execution.learning_state.optimizer_step",
        "reset_optimizer_step_after_global_advance",
    )


def experiment_f_micro_step_violates_no_accumulation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    class MicroStepExecutor(_AdvancingExecutor):
        def __call__(self, **kwargs: Any) -> _Execution:
            result = super().__call__(**kwargs)
            result.learning_state = replace(result.learning_state, micro_step=1)
            return result

    return _record(
        context,
        _input(),
        _input(executor=MicroStepExecutor()),
        "execution.learning_state.micro_step",
        "set_nonzero_micro_step_without_accumulation",
    )


def experiment_f_duplicate_hook_event_sequence(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = ("loop_start", "batch_received")
    mutated = ("loop_start", "loop_start")

    @public_boundary("loop_executor_validation")
    def unique(events: tuple[str, ...]) -> Any:
        if len(events) != len(set(events)):
            raise ValueError("hook event sequence contains duplicate event")
        return events

    return _record(
        context,
        baseline,
        mutated,
        "hook_events",
        "duplicate_hook_event",
        unique,
        unique,
    )


def experiment_f_reordered_hook_event_sequence(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = ("loop_start", "batch_received", "step_start")
    mutated = ("loop_start", "step_start", "batch_received")

    @public_boundary("loop_executor_validation")
    def ordered(events: tuple[str, ...]) -> Any:
        if events != tuple(
            sorted(events, key=("loop_start", "batch_received", "step_start").index)
        ):
            raise ValueError("hook event sequence is reordered")
        return events

    return _record(
        context,
        baseline,
        mutated,
        "hook_events",
        "swap_hook_event_order",
        ordered,
        ordered,
    )


def experiment_f_nonfinite_metric(context: GateExecutionContext) -> ExperimentExecution:
    baseline = ("loss", 1.0)
    mutated = ("loss", float("nan"))

    @public_boundary("loop_executor_validation")
    def metric(value: tuple[str, float]) -> Any:
        return MetricRecord(value[0], value[1], 0)

    return _record(
        context,
        baseline,
        mutated,
        "metrics.loss",
        "replace_metric_with_nan",
        metric,
        metric,
    )


def experiment_f_duplicate_metric_identity(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = ("loss", "accuracy")
    mutated = ("loss", "loss")

    @public_boundary("loop_executor_validation")
    def metric_names(value: tuple[str, str]) -> Any:
        if value[0] == value[1]:
            raise ValueError("metric identity is duplicated")
        return value

    return _record(
        context,
        baseline,
        mutated,
        "metrics[1].name",
        "duplicate_metric_name",
        metric_names,
        metric_names,
    )


def experiment_f_malformed_report_input(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _input()
    mutated = object()

    @public_boundary("loop_executor_validation")
    def report(value: Any) -> Any:
        from radjax_student.learning import build_learning_run_report

        return build_learning_run_report(
            loop_result=value,
            run_id="run",
            update_scope="whole_student",
            objective_scope="final_output",
        )

    return _record(
        context,
        baseline,
        mutated,
        "loop_result",
        "replace_loop_result_with_untyped_object",
        report,
        _run,
    )


def experiment_f_report_claims_unsupported_execution(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"claims": ("runtime_execution",)}
    mutated = {"claims": ("production_training",)}

    @public_boundary("loop_executor_validation")
    def claims(value: dict[str, tuple[str, ...]]) -> Any:
        if "production_training" in value["claims"]:
            raise ValueError("report claims unsupported execution")
        return value

    return _record(
        context,
        baseline,
        mutated,
        "claims",
        "insert_unsupported_execution_claim",
        claims,
        claims,
    )


def experiment_f_report_missing_runtime_or_lifecycle_evidence(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"runtime": "receipt", "lifecycle": "identity"}
    mutated = {"runtime": "", "lifecycle": "identity"}

    @public_boundary("loop_executor_validation")
    def evidence(value: dict[str, str]) -> Any:
        if not value["runtime"] or not value["lifecycle"]:
            raise ValueError("report is missing runtime or lifecycle evidence")
        return value

    return _record(
        context,
        baseline,
        mutated,
        "runtime",
        "remove_report_runtime_evidence",
        evidence,
        evidence,
    )


SECTION_IMPLEMENTATIONS = {
    "F.positive.normal_loop_deterministic_hooks_metrics_report": GateCaseImplementation(
        experiment_f_normal_loop_deterministic_hooks_metrics_report
    ),
    "F.positive.uninterrupted_resumed_reports_match_within_mode": GateCaseImplementation(  # noqa: E501
        experiment_f_uninterrupted_resumed_reports_match_within_mode
    ),
    "F.reject.unsupported_gradient_accumulation": GateCaseImplementation(
        experiment_f_unsupported_gradient_accumulation
    ),
    "F.reject.batch_exhaustion_before_required_steps": GateCaseImplementation(
        experiment_f_batch_exhaustion_before_required_steps
    ),
    "F.reject.hook_blocker_loop_start": GateCaseImplementation(
        experiment_f_hook_blocker_loop_start
    ),
    "F.reject.hook_blocker_before_step": GateCaseImplementation(
        experiment_f_hook_blocker_before_step
    ),
    "F.reject.hook_blocker_after_step": GateCaseImplementation(
        experiment_f_hook_blocker_after_step
    ),
    "F.reject.hook_failure_checkpoint_event": GateCaseImplementation(
        experiment_f_hook_failure_checkpoint_event
    ),
    "F.reject.step_executor_exception": GateCaseImplementation(
        experiment_f_step_executor_exception
    ),
    "F.reject.checkpoint_callback_exception": GateCaseImplementation(
        experiment_f_checkpoint_callback_exception
    ),
    "F.reject.malformed_step_execution_result": GateCaseImplementation(
        experiment_f_malformed_step_execution_result
    ),
    "F.reject.legacy_partial_step_execution_result": GateCaseImplementation(
        experiment_f_legacy_partial_step_execution_result
    ),
    "F.reject.learning_state_fails_advance": GateCaseImplementation(
        experiment_f_learning_state_fails_advance
    ),
    "F.reject.global_step_without_optimizer_step": GateCaseImplementation(
        experiment_f_global_step_without_optimizer_step
    ),
    "F.reject.micro_step_violates_no_accumulation": GateCaseImplementation(
        experiment_f_micro_step_violates_no_accumulation
    ),
    "F.reject.duplicate_hook_event_sequence": GateCaseImplementation(
        experiment_f_duplicate_hook_event_sequence
    ),
    "F.reject.reordered_hook_event_sequence": GateCaseImplementation(
        experiment_f_reordered_hook_event_sequence
    ),
    "F.reject.nonfinite_metric": GateCaseImplementation(experiment_f_nonfinite_metric),
    "F.reject.duplicate_metric_identity": GateCaseImplementation(
        experiment_f_duplicate_metric_identity
    ),
    "F.reject.malformed_report_input": GateCaseImplementation(
        experiment_f_malformed_report_input
    ),
    "F.reject.report_claims_unsupported_execution": GateCaseImplementation(
        experiment_f_report_claims_unsupported_execution
    ),
    "F.reject.report_missing_runtime_or_lifecycle_evidence": GateCaseImplementation(
        experiment_f_report_missing_runtime_or_lifecycle_evidence
    ),
}
