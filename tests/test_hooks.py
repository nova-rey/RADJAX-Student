from dataclasses import dataclass

import pytest

from radjax_student.learning import LearningIssue, MetricRecord
from radjax_student.learning.hooks import (
    HookContext,
    HookDispatchResult,
    HookExecutionReceipt,
    HookPolicy,
    HookRegistration,
    HookResult,
    dispatch_hooks,
    merge_core_and_hook_issues,
)


@dataclass(frozen=True)
class Hook:
    hook_id: str
    priority: int = 0
    result: object = HookResult()
    supported_events: tuple[str, ...] = ("loop_start",)

    def on_event(self, context):
        return self.result


def ctx():
    return HookContext("run", "loop_start", 1, 0)


def fail():
    return HookResult(
        "fail",
        warnings=(LearningIssue("explicit", "failure"),),
        metadata={"origin": "hook"},
    )


def test_01_valid_context():
    assert ctx().run_id == "run"


def test_02_empty_run_rejected():
    with pytest.raises(ValueError):
        HookContext("", "loop_start", 1, 0)


def test_03_event_rejected():
    with pytest.raises(ValueError):
        HookContext("r", "bad", 1, 0)


def test_04_negative_sequence_rejected():
    with pytest.raises(ValueError):
        HookContext("r", "loop_start", -1, 0)


def test_05_bool_sequence_rejected():
    with pytest.raises(ValueError):
        HookContext("r", "loop_start", True, 0)


def test_06_metadata_immutable():
    with pytest.raises(TypeError):
        ctx().metadata["x"] = 1


def test_07_result_status_rejected():
    with pytest.raises(ValueError):
        HookResult("bad")


def test_08_warning_requires_warning():
    with pytest.raises(ValueError):
        HookResult("warning")


def test_09_policy_rejected():
    with pytest.raises(ValueError):
        HookPolicy("bad")


def test_10_registration_enabled_rejected():
    with pytest.raises(ValueError):
        HookRegistration("h", 0, "yes")


def test_11_receipt_rejected():
    with pytest.raises(ValueError):
        HookExecutionReceipt("", "loop_start", 0, "pass")


def test_12_dispatch_result_rejected():
    with pytest.raises(ValueError):
        HookDispatchResult("pass", (), (), (LearningIssue("x", "x"),), (), ())


def test_13_duplicate_rejected():
    with pytest.raises(ValueError):
        dispatch_hooks((Hook("x"), Hook("x")), HookPolicy(), ctx())


def test_14_bad_hook_id_rejected():
    with pytest.raises(ValueError):
        dispatch_hooks((Hook(""),), HookPolicy(), ctx())


def test_15_bad_priority_rejected():
    with pytest.raises(ValueError):
        dispatch_hooks((Hook("x", True),), HookPolicy(), ctx())


def test_16_bad_events_rejected():
    with pytest.raises(ValueError):
        dispatch_hooks((Hook("x", supported_events=("bad",)),), HookPolicy(), ctx())


def test_17_order_priority():
    assert [
        x.hook_id
        for x in dispatch_hooks(
            (Hook("b", 2), Hook("a", 1)), HookPolicy(), ctx()
        ).receipts
    ] == ["a", "b"]


def test_18_order_lexical():
    assert [
        x.hook_id
        for x in dispatch_hooks((Hook("b"), Hook("a")), HookPolicy(), ctx()).receipts
    ] == ["a", "b"]


def test_19_unsupported_skipped():
    assert not dispatch_hooks(
        (Hook("x", supported_events=("failure",)),), HookPolicy(), ctx()
    ).receipts


def test_20_disabled_skipped():
    assert not dispatch_hooks((Hook("x"),), HookPolicy(), ctx(), ("x",)).receipts


def test_21_metric_emitted():
    assert dispatch_hooks(
        (Hook("x", result=HookResult(metrics=(MetricRecord("loss", 1, 0),))),),
        HookPolicy(),
        ctx(),
    ).metrics


def test_22_warning_emitted():
    assert dispatch_hooks(
        (Hook("x", result=HookResult("warning", warnings=(LearningIssue("w", "w"),))),),
        HookPolicy(),
        ctx(),
    ).warnings


def test_23_explicit_failure_details():
    assert (
        dispatch_hooks((Hook("x", result=fail()),), HookPolicy(), ctx())
        .receipts[0]
        .metadata["origin"]
        == "hook"
    )


def test_24_fail_fast():
    assert (
        dispatch_hooks(
            (Hook("x", result=fail()), Hook("z")), HookPolicy(), ctx()
        ).status
        == "fail"
    )


def test_25_continue_invalid_code():
    assert (
        dispatch_hooks(
            (Hook("x", result=object()),), HookPolicy("warn_and_continue"), ctx()
        )
        .warnings[-1]
        .code
        == "learning_hook_result_invalid"
    )


def test_26_disable_invalid_code():
    assert dispatch_hooks(
        (Hook("x", result=object()),), HookPolicy("disable_hook"), ctx()
    ).disabled_hook_ids == ("x",)


def test_27_disabled_persists():
    assert not dispatch_hooks((Hook("x"),), HookPolicy(), ctx(), ("x",)).receipts


def test_28_metric_policy_code():
    assert (
        dispatch_hooks(
            (Hook("x", result=HookResult(metrics=(MetricRecord("loss", 1, 0),))),),
            HookPolicy(allow_metric_emission=False),
            ctx(),
        )
        .blockers[0]
        .code
        == "learning_hook_metric_policy_violation"
    )


def test_29_merge_preserves_core():
    assert [
        x.code
        for x in merge_core_and_hook_issues(
            (LearningIssue("core", "c"),),
            dispatch_hooks((Hook("x", result=fail()),), HookPolicy(), ctx()),
        )
    ] == ["core", "learning_hook_failed"]


def test_30_serialization_deterministic():
    assert (
        ctx().to_dict() == ctx().to_dict()
        and HookPolicy().to_dict() == HookPolicy().to_dict()
    )
