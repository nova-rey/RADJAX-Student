from dataclasses import dataclass, field

from radjax_student.architecture import ArchitectureConfig
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.learning import (
    HookPolicy,
    HookResult,
    LearningBatch,
    LearningIssue,
    LearningState,
    MetricRecord,
)
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
from tests.test_single_learning_step import LinearObjective

DEFAULT_POLICY = HookPolicy()


@dataclass
class RecordingHook:
    hook_id: str = "record"
    priority: int = 0
    supported_events: tuple[str, ...] = (
        "loop_start",
        "batch_received",
        "step_start",
        "step_end",
        "checkpoint",
        "loop_end",
        "failure",
    )
    events: list = field(default_factory=list)
    result: HookResult = HookResult()

    def on_event(self, c):
        self.events.append(
            (c.event_type, c.event_sequence, c.global_step, dict(c.metadata))
        )
        return self.result


@dataclass(frozen=True)
class FailingHook:
    hook_id: str = "fail"
    priority: int = 0
    supported_events: tuple[str, ...] = (
        "loop_start",
        "batch_received",
        "step_start",
        "step_end",
        "checkpoint",
        "loop_end",
        "failure",
    )
    event_to_fail: str = "loop_start"

    def on_event(self, c):
        return (
            HookResult("fail", warnings=(LearningIssue("hook_test", "failed"),))
            if c.event_type == self.event_to_fail
            else HookResult()
        )


@dataclass
class CountingBatchSource(SyntheticBatchSource):
    next_calls: int = 0

    def next_batch(self):
        self.next_calls += 1
        return super().next_batch()


@dataclass
class DisableCountingHook:
    hook_id: str = "disable"
    priority: int = 0
    supported_events: tuple[str, ...] = ("batch_received",)
    calls: int = 0

    def on_event(self, context):
        self.calls += 1
        return HookResult("fail", warnings=(LearningIssue("detail", "d"),))


def build(hooks=(), policy=DEFAULT_POLICY, checkpoint=None, steps=1):
    arch = FakeArchitecturePlugin()
    cat = arch.describe_parameters()
    opt = SgdOptimizer()
    cfg = OptimizerConfig(optimizer_id="sgd.v1", learning_rate=0.1)
    state = opt.initialize_state(
        OptimizerInitRequest(
            cfg,
            cat,
            arch.resolve_update_scope(
                LearningState(run_id="r").active_update_scope, cat
            ),
        )
    ).optimizer_state
    batch = LearningBatch(
        batch_id="b",
        inputs={"token_ids": {"rank": 2, "sequence_length": 1, "x": 1.0}},
        targets={"target": {"y": 3.0}},
    )
    return run_learning_loop(
        config=LearningLoopConfig(
            max_steps=steps, checkpoint_every_n_steps=1 if checkpoint else None
        ),
        architecture=arch,
        architecture_config=ArchitectureConfig(
            architecture_id=arch.architecture_id, sequence_length=4
        ),
        optimizer=opt,
        optimizer_config=cfg,
        optimizer_state=state,
        learning_state=LearningState(run_id="r"),
        parameters={"head.weight": 0.0, "trunk.bias": 0.0, "trunk.weight": 0.0},
        objective=LinearObjective(),
        batch_source=SyntheticBatchSource((batch,) * 3),
        checkpoint=checkpoint,
        hooks=hooks,
        hook_policy=policy,
    )


def test_normal_max_steps_lifecycle_order():
    h = RecordingHook()
    assert build((h,), steps=1).hook_events == (
        "loop_start",
        "batch_received",
        "step_start",
        "step_end",
        "loop_end",
    )


def test_successful_checkpoint_event_order():
    h = RecordingHook()
    assert build((h,), checkpoint=lambda e: "c").hook_events[-2:] == (
        "checkpoint",
        "loop_end",
    )


def test_source_exhaustion_emits_loop_end():
    h = RecordingHook()
    assert build((h,), steps=4).hook_events[-1] == "loop_end"


def test_learning_step_exception_emits_failure(monkeypatch):
    monkeypatch.setattr(
        "radjax_student.steps.loop.learning_step",
        lambda **k: (_ for _ in ()).throw(RuntimeError()),
    )
    h = RecordingHook()
    assert build((h,)).hook_events[-1] == "failure"


def test_learning_step_exception_does_not_emit_step_end(monkeypatch):
    monkeypatch.setattr(
        "radjax_student.steps.loop.learning_step",
        lambda **k: (_ for _ in ()).throw(RuntimeError()),
    )
    assert "step_end" not in build().hook_events


def test_checkpoint_exception_emits_failure():
    h = RecordingHook()
    assert (
        build(
            (h,), checkpoint=lambda e: (_ for _ in ()).throw(RuntimeError())
        ).hook_events[-1]
        == "failure"
    )


def test_checkpoint_exception_does_not_emit_checkpoint():
    assert (
        "checkpoint"
        not in build(
            checkpoint=lambda e: (_ for _ in ()).throw(RuntimeError())
        ).hook_events
    )


def test_failure_context_contains_stage_and_exception_type(monkeypatch):
    monkeypatch.setattr(
        "radjax_student.steps.loop.learning_step",
        lambda **k: (_ for _ in ()).throw(RuntimeError()),
    )
    h = RecordingHook()
    build((h,))
    assert (
        h.events[-1][3]["failure_stage"] == "learning_step"
        and h.events[-1][3]["exception_type"] == "RuntimeError"
    )


def test_loop_start_fail_fast_returns_hook_failure():
    assert build((FailingHook(),)).stop_reason == "hook_failure"


def test_step_start_fail_fast_skips_learning_step():
    assert build((FailingHook(event_to_fail="step_start"),)).steps_completed == 0


def test_step_end_fail_fast_stops():
    assert build((FailingHook(event_to_fail="step_end"),)).stop_reason == "hook_failure"


def test_checkpoint_fail_fast_returns_hook_failure():
    assert (
        build(
            (FailingHook(event_to_fail="checkpoint"),), checkpoint=lambda e: "c"
        ).stop_reason
        == "hook_failure"
    )


def test_checkpoint_fail_fast_prevents_later_steps():
    assert (
        build(
            (FailingHook(event_to_fail="checkpoint"),),
            checkpoint=lambda e: "c",
            steps=2,
        ).steps_completed
        == 1
    )


def test_loop_end_fail_fast_returns_hook_failure():
    assert build((FailingHook(event_to_fail="loop_end"),)).stop_reason == "hook_failure"


def test_warn_and_continue_completes_learning():
    assert (
        build(
            (FailingHook(event_to_fail="step_start"),), HookPolicy("warn_and_continue")
        ).status
        == "pass"
    )


def test_warn_and_continue_preserves_warning():
    assert build(
        (FailingHook(event_to_fail="step_start"),), HookPolicy("warn_and_continue")
    ).warnings


def test_hook_metric_appears_in_loop_metrics():
    h = RecordingHook(result=HookResult(metrics=(MetricRecord("hook.metric", 1, 0),)))
    assert "hook.metric" in [m.name for m in build((h,)).metrics]


def test_hook_warning_appears_in_loop_warnings():
    h = RecordingHook(
        result=HookResult("warning", warnings=(LearningIssue("hook.warning", "w"),))
    )
    assert "hook.warning" in [w.code for w in build((h,)).warnings]


def test_learning_step_core_reason_survives_failure_hook_blocker(monkeypatch):
    monkeypatch.setattr(
        "radjax_student.steps.loop.learning_step",
        lambda **k: (_ for _ in ()).throw(RuntimeError()),
    )
    r = build((FailingHook(event_to_fail="failure"),))
    assert r.stop_reason == "learning_step_failure" and r.hook_blockers


def test_checkpoint_core_reason_survives_failure_hook_blocker():
    r = build(
        (FailingHook(event_to_fail="failure"),),
        checkpoint=lambda e: (_ for _ in ()).throw(RuntimeError()),
    )
    assert r.stop_reason == "checkpoint_failure" and r.hook_blockers


def test_identical_runs_have_identical_event_order():
    assert (
        build((RecordingHook(),)).hook_events == build((RecordingHook(),)).hook_events
    )


def test_loop_start_fail_fast_consumes_no_batch(monkeypatch):
    source = CountingBatchSource(())
    # Build through the real loop but replace only the source factory argument.
    monkeypatch.setattr(
        "radjax_student.steps.loop.SyntheticBatchSource", lambda *a: source
    )
    assert (
        build((FailingHook(),)).status == "fail"
        and source.next_calls == 0
        and source.position == 0
    )


def test_step_start_fail_fast_does_not_call_learning_step(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "radjax_student.steps.loop.learning_step", lambda **k: calls.append(k) or None
    )
    result = build((FailingHook(event_to_fail="step_start"),))
    assert (
        result.stop_reason == "hook_failure"
        and calls == []
        and result.steps_completed == 0
    )


def test_step_end_fail_fast_skips_checkpoint():
    calls = []
    result = build(
        (FailingHook(event_to_fail="step_end"),),
        checkpoint=lambda e: calls.append(e) or "c",
    )
    assert (
        result.stop_reason == "hook_failure"
        and calls == []
        and result.steps_completed == 1
    )


def test_disable_hook_runs_once_then_remains_disabled():
    hook = DisableCountingHook()
    result = build((hook,), HookPolicy("disable_hook"), steps=2)
    assert (
        hook.calls == 1
        and result.status == "pass"
        and result.steps_completed == 2
        and "learning_hook_disabled" in [x.code for x in result.warnings]
    )


def test_failure_hook_blocker_code_preserved(monkeypatch):
    monkeypatch.setattr(
        "radjax_student.steps.loop.learning_step",
        lambda **k: (_ for _ in ()).throw(RuntimeError()),
    )
    assert [
        x.code for x in build((FailingHook(event_to_fail="failure"),)).hook_blockers
    ] == ["learning_hook_failed"]


def test_failure_hook_warning_preserved(monkeypatch):
    monkeypatch.setattr(
        "radjax_student.steps.loop.learning_step",
        lambda **k: (_ for _ in ()).throw(RuntimeError()),
    )
    assert "hook_test" in [
        x.code for x in build((FailingHook(event_to_fail="failure"),)).warnings
    ]


def test_event_sequence_numbers_are_monotonic():
    hook = RecordingHook()
    build((hook,), steps=2)
    seq = [x[1] for x in hook.events]
    assert seq == sorted(seq) == list(range(1, len(seq) + 1)) and len(set(seq)) == len(
        seq
    )


def test_hook_context_exposes_no_core_mutable_state():
    hook = RecordingHook()
    build((hook,))
    names = set(
        __import__(
            "radjax_student.learning.hooks", fromlist=["HookContext"]
        ).HookContext.__dataclass_fields__
    )
    assert not names & {
        "parameters",
        "gradients",
        "optimizer_state",
        "architecture_state",
        "runtime_state",
        "checkpoint_payload",
        "update_scope",
        "objective_scope",
    }


def test_identical_runs_preserve_event_sequence_and_order():
    a, b = RecordingHook(), RecordingHook()
    build((a,), steps=2)
    build((b,), steps=2)
    assert [(x[0], x[1], x[2]) for x in a.events] == [
        (x[0], x[1], x[2]) for x in b.events
    ]


def test_disable_hook_preserves_failure_class_and_disablement():
    result = build((DisableCountingHook(),), HookPolicy("disable_hook"), steps=2)
    codes = [x.code for x in result.warnings]
    assert (
        "learning_hook_failed_continue" in codes and "learning_hook_disabled" in codes
    )
