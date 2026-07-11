from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from radjax_student.architecture import ArchitectureConfig
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.learning import (
    HookResult,
    LearningBatch,
    LearningIssue,
    LearningState,
    MetricRecord,
    RunCheckpointSummary,
    RunIssueSummary,
    RunLifecycleSummary,
    RunMetricSummary,
    RunStatusSummary,
    build_learning_run_report,
)
from radjax_student.learning.run_report import CLAIMS, NON_CLAIMS, LearningRunReport
from radjax_student.optimizers import (
    OptimizerConfig,
    OptimizerInitRequest,
    SgdOptimizer,
)
from radjax_student.steps import (
    LearningLoopConfig,
    SyntheticBatchSource,
    run_learning_loop,
)
from radjax_student.steps.loop import LearningLoopResult


@dataclass(frozen=True)
class LinearObjective:
    def evaluate(self, parameters, batch):
        x = float(batch.inputs["token_ids"]["x"])
        target = float(batch.targets["target"]["y"])
        error = parameters["trunk.weight"] * x + parameters["head.weight"] - target
        return error * error, {
            "head.weight": 2.0 * error,
            "trunk.bias": 0.0,
            "trunk.weight": 2.0 * error * x,
        }


@dataclass(frozen=True)
class FailingHook:
    hook_id: str = "report-fail"
    priority: int = 0
    supported_events: tuple[str, ...] = (
        "loop_start",
        "batch_received",
        "step_start",
        "step_end",
        "checkpoint",
        "loop_end",
    )
    event_to_fail: str = "loop_start"

    def on_event(self, context):
        if context.event_type == self.event_to_fail:
            return HookResult("fail", warnings=(LearningIssue("hook", "failed"),))
        return HookResult()


def make_loop_result(status="pass", steps_completed=2, global_step=2):
    return LearningLoopResult(
        status=status,
        final_execution=None,
        steps_completed=steps_completed,
        global_step=global_step,
        batches_consumed=2,
        stop_reason="max_steps",
        metrics=(
            MetricRecord("loss", 3, 1),
            MetricRecord("loss", 1, 2),
            MetricRecord("learning_rate", 0.1, 2),
        ),
        checkpoints=("receipt-one", "receipt-two"),
        warnings=(
            LearningIssue("warning-one", "one"),
            LearningIssue("warning-one", "two"),
        ),
        hook_events=("loop_start", "step_end", "loop_end"),
        hook_blockers=(
            LearningIssue("blocker-one", "one"),
            LearningIssue("blocker-two", "two"),
        ),
    )


def make_report(**kwargs):
    defaults = {
        "loop_result": make_loop_result(),
        "run_id": "report-run",
        "update_scope": "whole_student",
        "objective_scope": "final_output",
    }
    defaults.update(kwargs)
    return build_learning_run_report(**defaults)


def run_loop(
    *,
    emit_run_report=False,
    max_steps=1,
    starting_global_step=0,
    hooks=(),
    checkpoint=None,
    source_length=3,
):
    architecture = FakeArchitecturePlugin()
    optimizer = SgdOptimizer()
    optimizer_config = OptimizerConfig(
        optimizer_id=optimizer.optimizer_id, learning_rate=0.1
    )
    state = LearningState(run_id="loop-report", global_step=starting_global_step)
    selection = architecture.resolve_update_scope(
        state.active_update_scope, architecture.describe_parameters()
    )
    optimizer_state = optimizer.initialize_state(
        OptimizerInitRequest(
            optimizer_config, architecture.describe_parameters(), selection
        )
    ).optimizer_state
    batch = LearningBatch(
        batch_id="batch",
        inputs={"token_ids": {"rank": 2, "sequence_length": 1, "x": 1.0}},
        targets={"target": {"y": 3.0}},
    )
    return run_learning_loop(
        config=LearningLoopConfig(
            max_steps=max_steps,
            checkpoint_every_n_steps=1 if checkpoint is not None else None,
        ),
        architecture=architecture,
        architecture_config=ArchitectureConfig(
            architecture_id=architecture.architecture_id, sequence_length=4
        ),
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        learning_state=state,
        parameters={"head.weight": 0.0, "trunk.bias": 0.0, "trunk.weight": 0.0},
        objective=LinearObjective(),
        batch_source=SyntheticBatchSource((batch,) * source_length),
        checkpoint=checkpoint,
        hooks=hooks,
        emit_run_report=emit_run_report,
    )


def test_01_valid_status_summary():
    assert RunStatusSummary("pass", "max_steps", 2, 2).to_dict()["status"] == "pass"


def test_02_invalid_status_rejected():
    with pytest.raises(ValueError):
        RunStatusSummary("unknown", "max_steps", 0, 0)


def test_03_negative_steps_rejected():
    with pytest.raises(ValueError):
        RunStatusSummary("pass", "max_steps", -1, 0)


def test_04_valid_metric_summary():
    assert RunMetricSummary("loss", 1, 1, 1, 1, 1, 1).sum == 1.0


def test_05_zero_metric_count_rejected():
    with pytest.raises(ValueError):
        RunMetricSummary("loss", 0, 1, 1, 1, 1, 1)


def test_06_nonfinite_metric_rejected():
    with pytest.raises(ValueError):
        RunMetricSummary("loss", 1, float("nan"), 1, 1, 1, 1)


def test_07_lifecycle_count_mismatch_rejected():
    with pytest.raises(ValueError):
        RunLifecycleSummary(("loop_start",), 2, "loop_start", "loop_start")


def test_08_invalid_lifecycle_event_rejected():
    with pytest.raises(ValueError):
        RunLifecycleSummary(("unknown",), 1, "unknown", "unknown")


def test_09_issue_count_mismatch_rejected():
    with pytest.raises(ValueError):
        RunIssueSummary(("warning",), (), 0, 0)


def test_10_empty_checkpoint_receipt_rejected():
    with pytest.raises(ValueError):
        RunCheckpointSummary(("",))


def test_11_invalid_run_id_rejected():
    with pytest.raises(ValueError):
        make_report(run_id="")


def test_12_invalid_schema_version_rejected():
    report = make_report()
    with pytest.raises(ValueError):
        LearningRunReport(
            report.run_id,
            report.status,
            report.metrics,
            report.lifecycle,
            report.issues,
            report.checkpoints,
            report.scopes,
            schema_version="wrong",
        )


def test_13_duplicate_claims_rejected():
    report = make_report()
    with pytest.raises(ValueError):
        LearningRunReport(
            report.run_id,
            report.status,
            report.metrics,
            report.lifecycle,
            report.issues,
            report.checkpoints,
            report.scopes,
            claims_made=CLAIMS + (CLAIMS[0],),
        )


def test_14_invalid_metadata_rejected():
    with pytest.raises(ValueError):
        make_report(metadata={"parameters": {"raw": 1}})


def test_15_successful_loop_result_converts():
    assert make_report().status.status == "pass"


def test_16_failed_loop_result_converts():
    assert make_report(loop_result=make_loop_result("fail")).status.status == "fail"


def test_17_warning_codes_preserved_in_order():
    assert make_report().issues.warning_codes == ("warning-one", "warning-one")


def test_18_hook_blocker_codes_preserved_in_order():
    assert make_report().issues.hook_blocker_codes == ("blocker-one", "blocker-two")


def test_19_checkpoint_receipts_preserved_in_order():
    assert make_report().checkpoints.receipts == ("receipt-one", "receipt-two")


def test_20_lifecycle_events_preserved_in_order():
    assert make_report().lifecycle.events == ("loop_start", "step_end", "loop_end")


def test_21_update_scope_preserved():
    assert (
        make_report(update_scope="named_region").scopes.update_scope == "named_region"
    )


def test_22_objective_scope_preserved():
    assert (
        make_report(objective_scope="intermediate_surface").scopes.objective_scope
        == "intermediate_surface"
    )


def test_23_run_id_preserved():
    assert make_report(run_id="keep-me").run_id == "keep-me"


def test_24_stop_reason_preserved():
    assert make_report().status.stop_reason == "max_steps"


def test_25_steps_completed_preserved():
    assert make_report().status.steps_completed == 2


def test_26_global_step_preserved():
    assert make_report().status.global_step == 2


def test_27_metrics_grouped_by_name():
    assert len(make_report().metrics) == 2


def test_28_metric_names_ordered_deterministically():
    assert [metric.name for metric in make_report().metrics] == [
        "learning_rate",
        "loss",
    ]


def test_29_metric_count_correct():
    assert make_report().metrics[1].count == 2


def test_30_metric_last_correct():
    assert make_report().metrics[1].last == 1.0


def test_31_metric_minimum_correct():
    assert make_report().metrics[1].minimum == 1.0


def test_32_metric_maximum_correct():
    assert make_report().metrics[1].maximum == 3.0


def test_33_metric_mean_correct():
    assert make_report().metrics[1].mean == 2.0


def test_34_metric_sum_correct():
    assert make_report().metrics[1].sum == 4.0


def test_35_nonfinite_input_rejected():
    with pytest.raises(ValueError):
        MetricRecord("loss", float("inf"), 0)


def test_36_bounded_history_source_identified_honestly():
    assert make_report().to_dict()["metric_summary_source"] == "bounded_history"


def test_37_to_dict_deterministic():
    assert make_report().to_dict() == make_report().to_dict()


def test_38_to_json_deterministic():
    assert make_report().to_json() == make_report().to_json()


def test_39_json_keys_sorted():
    assert make_report().to_json() == json.dumps(
        make_report().to_dict(), sort_keys=True, separators=(",", ":")
    )


def test_40_no_raw_parameter_tree_in_json():
    assert '"parameters"' not in make_report().to_json()


def test_41_no_optimizer_state_in_json():
    assert '"optimizer_state"' not in make_report().to_json()


def test_42_no_runtime_handle_in_json():
    assert '"runtime_handle"' not in make_report().to_json()


def test_43_no_traceback_text_in_json():
    assert '"traceback"' not in make_report().to_json()


def test_44_claims_made_exact():
    assert make_report().claims_made == CLAIMS


def test_45_claims_not_made_include_required_set():
    assert set(NON_CLAIMS).issubset(make_report().claims_not_made)


def test_46_default_loop_result_has_no_report():
    assert run_loop().report is None


def test_47_opt_in_loop_result_has_report():
    assert run_loop(emit_run_report=True).report is not None


def test_48_reporting_does_not_change_steps_completed():
    assert run_loop().steps_completed == run_loop(emit_run_report=True).steps_completed


def test_49_reporting_does_not_change_stop_reason():
    assert run_loop().stop_reason == run_loop(emit_run_report=True).stop_reason


def test_50_reporting_does_not_change_parameters_or_optimizer_outcome():
    plain = run_loop()
    reported = run_loop(emit_run_report=True)
    assert plain.final_execution.parameters == reported.final_execution.parameters
    assert (
        plain.final_execution.optimizer_state
        == reported.final_execution.optimizer_state
    )


def test_51_report_construction_occurs_after_loop_completion(monkeypatch):
    observed = []
    original = build_learning_run_report

    def observing_builder(**kwargs):
        observed.append(kwargs["loop_result"].steps_completed)
        return original(**kwargs)

    monkeypatch.setattr(
        "radjax_student.learning.run_report.build_learning_run_report",
        observing_builder,
    )
    result = run_loop(emit_run_report=True)
    assert observed == [result.steps_completed] and result.report is not None


def test_52_report_failure_does_not_rewrite_successful_learning_outcome(monkeypatch):
    monkeypatch.setattr(
        "radjax_student.learning.run_report.build_learning_run_report",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("report failure")),
    )
    result = run_loop(emit_run_report=True)
    assert (
        result.status == "pass"
        and result.stop_reason == "max_steps"
        and result.report is None
    )


def test_resumed_run_report_preserves_actual_global_step():
    report = make_report(
        loop_result=make_loop_result(steps_completed=5, global_step=105)
    )
    assert report.status.steps_completed == 5 and report.status.global_step == 105


def test_report_global_step_is_not_derived_from_steps_completed():
    assert (
        make_report(
            loop_result=make_loop_result(steps_completed=2, global_step=50)
        ).status.global_step
        == 50
    )


def test_fresh_run_report_global_step_matches_final_state():
    report = make_report(loop_result=make_loop_result(steps_completed=2, global_step=2))
    assert report.status.steps_completed == 2 and report.status.global_step == 2


def test_loop_start_hook_blocker_appears_in_report():
    result = run_loop(emit_run_report=True, hooks=(FailingHook(),))
    assert (
        result.stop_reason == "hook_failure"
        and result.hook_blockers[0].code == "learning_hook_failed"
        and result.report.issues.hook_blocker_codes == ("learning_hook_failed",)
    )


def test_step_end_hook_blocker_appears_in_report():
    result = run_loop(
        emit_run_report=True, hooks=(FailingHook(event_to_fail="step_end"),)
    )
    assert (
        result.steps_completed == 1
        and result.global_step == 1
        and result.report.issues.hook_blocker_codes == ("learning_hook_failed",)
    )


def test_checkpoint_hook_blocker_appears_in_report():
    result = run_loop(
        emit_run_report=True,
        hooks=(FailingHook(event_to_fail="checkpoint"),),
        checkpoint=lambda execution: "receipt",
    )
    assert (
        result.checkpoints == ("receipt",)
        and result.stop_reason == "hook_failure"
        and result.report.issues.hook_blocker_codes == ("learning_hook_failed",)
    )


def test_loop_end_hook_blocker_appears_in_report():
    result = run_loop(
        emit_run_report=True, hooks=(FailingHook(event_to_fail="loop_end"),)
    )
    assert (
        result.stop_reason == "hook_failure"
        and result.report.issues.hook_blocker_codes == ("learning_hook_failed",)
    )


def test_hook_blocker_order_preserved_in_report():
    assert make_report().issues.hook_blocker_codes == ("blocker-one", "blocker-two")
