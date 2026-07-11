from dataclasses import FrozenInstanceError, dataclass

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


def test_31_fail_fast_preserves_warning():
    assert "explicit" in [
        x.code
        for x in dispatch_hooks(
            (Hook("x", result=fail()),), HookPolicy(), ctx()
        ).warnings
    ]


def test_32_fail_fast_stops_later():
    assert (
        len(
            dispatch_hooks(
                (Hook("x", result=fail()), Hook("z")), HookPolicy(), ctx()
            ).receipts
        )
        == 1
    )


def test_33_disable_invalid_class():
    assert "learning_hook_result_invalid" in [
        x.code
        for x in dispatch_hooks(
            (Hook("x", result=object()),), HookPolicy("disable_hook"), ctx()
        ).warnings
    ]


def test_34_disable_invalid_disabled():
    assert "learning_hook_disabled" in [
        x.code
        for x in dispatch_hooks(
            (Hook("x", result=object()),), HookPolicy("disable_hook"), ctx()
        ).warnings
    ]


def test_35_disable_metric_class():
    assert "learning_hook_metric_policy_violation" in [
        x.code
        for x in dispatch_hooks(
            (Hook("x", result=HookResult(metrics=(MetricRecord("loss", 1, 0),))),),
            HookPolicy("disable_hook", False),
            ctx(),
        ).warnings
    ]


def test_36_disable_metric_disabled():
    assert "learning_hook_disabled" in [
        x.code
        for x in dispatch_hooks(
            (Hook("x", result=HookResult(metrics=(MetricRecord("loss", 1, 0),))),),
            HookPolicy("disable_hook", False),
            ctx(),
        ).warnings
    ]


@dataclass(frozen=True)
class RaisingHook:
    hook_id: str = "raise"
    priority: int = 0
    supported_events: tuple[str, ...] = ("loop_start",)

    def on_event(self, context):
        raise RuntimeError("bad")


def test_37_exception_type():
    assert (
        dispatch_hooks((RaisingHook(),), HookPolicy(), ctx())
        .receipts[0]
        .metadata["exception_type"]
        == "RuntimeError"
    )


def test_38_disable_exception_class():
    assert "learning_hook_failed_continue" in [
        x.code
        for x in dispatch_hooks(
            (RaisingHook(),), HookPolicy("disable_hook"), ctx()
        ).warnings
    ]


def test_39_disable_exception_disabled():
    assert "learning_hook_disabled" in [
        x.code
        for x in dispatch_hooks(
            (RaisingHook(),), HookPolicy("disable_hook"), ctx()
        ).warnings
    ]


def test_40_receipt_original_code():
    assert (
        dispatch_hooks((Hook("x", result=object()),), HookPolicy("disable_hook"), ctx())
        .receipts[0]
        .failure_code
        == "learning_hook_result_invalid"
    )


def test_41_receipt_disabled():
    assert (
        dispatch_hooks((Hook("x", result=object()),), HookPolicy("disable_hook"), ctx())
        .receipts[0]
        .disabled_after_failure
    )


def test_42_context_immutable_field():
    with pytest.raises(FrozenInstanceError):
        ctx().run_id = "changed"


def test_43_metrics_tuple_immutable():
    with pytest.raises(FrozenInstanceError):
        ctx().metrics += ()


def test_44_registration_serialization():
    assert HookRegistration("x", 1).to_dict() == HookRegistration("x", 1).to_dict()


def test_45_dispatch_serialization():
    r = dispatch_hooks((Hook("x"),), HookPolicy(), ctx())
    assert r.to_dict() == r.to_dict()
