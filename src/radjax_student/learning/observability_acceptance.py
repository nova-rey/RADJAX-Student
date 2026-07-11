"""Deterministic in-memory P3.8 observability acceptance audit."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.learning._json import (
    freeze_json_mapping,
    json_value,
    unique_strings,
)
from radjax_student.learning.errors import LearningIssue
from radjax_student.learning.hooks import (
    HookContext,
    HookPolicy,
    HookResult,
    dispatch_hooks,
)
from radjax_student.learning.models import LearningBatch, LearningState, MetricRecord
from radjax_student.learning.telemetry import (
    MetricRetentionPolicy,
    MetricSeries,
    canonical_telemetry_json,
)

SCHEMA = "radjax.p3_8_observability_acceptance.v1"
CLAIMS = (
    "p3_8_metrics_contract_validated",
    "p3_8_hook_contract_validated",
    "p3_8_loop_integration_validated",
    "p3_8_run_reporting_validated",
    "p3_8_deterministic_observability_validated",
)
NON_CLAIMS = (
    "model_quality",
    "real_architecture_support",
    "tome_training",
    "language_modeling",
    "distributed_training",
    "accelerator_performance",
    "external_telemetry",
    "evaluation",
    "p3_9_synthetic_learning_complete",
)
_SECTION_CODES = {
    "metrics_contract_valid": "p3_8_metrics_contract_failed",
    "hook_contract_valid": "p3_8_hook_contract_failed",
    "loop_integration_valid": "p3_8_loop_integration_failed",
    "run_reporting_valid": "p3_8_run_reporting_failed",
    "deterministic_replay_valid": "p3_8_deterministic_replay_failed",
    "failure_paths_valid": "p3_8_failure_paths_failed",
    "observer_only_boundary_valid": "p3_8_observer_boundary_failed",
    "bounded_history_claim_valid": "p3_8_bounded_history_claim_failed",
    "import_boundary_valid": "p3_8_import_boundary_failed",
    "documentation_valid": "p3_8_documentation_failed",
    "test_inventory_valid": "p3_8_test_inventory_failed",
}
_FORBIDDEN_METADATA = {
    "parameters",
    "gradients",
    "optimizer_state",
    "architecture_state",
    "runtime_handle",
    "raw_batch",
    "traceback",
}


@dataclass(frozen=True)
class P38ObservabilityAcceptanceReceipt:
    schema_version: str
    status: Literal["pass", "fail"]
    metrics_contract_valid: bool
    hook_contract_valid: bool
    loop_integration_valid: bool
    run_reporting_valid: bool
    deterministic_replay_valid: bool
    failure_paths_valid: bool
    observer_only_boundary_valid: bool
    bounded_history_claim_valid: bool
    import_boundary_valid: bool
    documentation_valid: bool
    test_inventory_valid: bool
    blockers: tuple[LearningIssue, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    claims_made: tuple[str, ...] = CLAIMS
    claims_not_made: tuple[str, ...] = NON_CLAIMS
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA or self.status not in ("pass", "fail"):
            raise ValueError("invalid P3.8 observability acceptance receipt")
        flags = _receipt_flags(self)
        if any(type(value) is not bool for value in flags.values()):
            raise TypeError("acceptance validity flags must be booleans")
        blockers, warnings = tuple(self.blockers), tuple(self.warnings)
        if any(
            not isinstance(issue, LearningIssue) for issue in (*blockers, *warnings)
        ):
            raise TypeError("acceptance issues must be LearningIssue values")
        if self.status == "pass" and (not all(flags.values()) or blockers):
            raise ValueError("passing receipt requires valid sections and no blockers")
        if self.status == "fail" and all(flags.values()) and not blockers:
            raise ValueError("failing receipt requires a failed section or blocker")
        claims_made = unique_strings(self.claims_made, "claims_made")
        claims_not_made = unique_strings(self.claims_not_made, "claims_not_made")
        if claims_made != CLAIMS or not set(NON_CLAIMS).issubset(claims_not_made):
            raise ValueError("acceptance claims are invalid")
        _validate_safe_metadata(self.metadata)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_made", claims_made)
        object.__setattr__(self, "claims_not_made", claims_not_made)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            **_receipt_flags(self),
            "blockers": [issue.to_dict() for issue in self.blockers],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "claims_made": list(self.claims_made),
            "claims_not_made": list(self.claims_not_made),
            "metadata": json_value(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def run_p3_8_observability_acceptance() -> P38ObservabilityAcceptanceReceipt:
    """Audit the completed P3.8 stack without changing learning behavior."""

    audits: tuple[tuple[str, Callable[[], bool]], ...] = (
        ("metrics_contract_valid", _audit_metrics),
        ("hook_contract_valid", _audit_hooks),
        ("loop_integration_valid", _audit_loop_integration),
        ("run_reporting_valid", _audit_run_reporting),
        ("deterministic_replay_valid", _audit_deterministic_replay),
        ("failure_paths_valid", _audit_failure_paths),
        ("observer_only_boundary_valid", _audit_observer_only_boundary),
        ("bounded_history_claim_valid", _audit_bounded_history),
        ("import_boundary_valid", _audit_import_boundary),
        ("documentation_valid", _audit_documentation),
        ("test_inventory_valid", _audit_test_inventory),
    )
    values: dict[str, bool] = {}
    blockers: list[LearningIssue] = []
    for field_name, audit in audits:
        try:
            values[field_name] = audit()
        except (TypeError, ValueError, AttributeError, KeyError):
            values[field_name] = False
        if not values[field_name]:
            blockers.append(
                LearningIssue(
                    _SECTION_CODES[field_name],
                    "P3.8 acceptance audit failed",
                    {
                        "section": field_name,
                        "check": field_name,
                        "expected": True,
                        "actual": values[field_name],
                    },
                )
            )
    return _receipt_from_values(values, tuple(blockers))


def _receipt_from_values(
    values: Mapping[str, bool], blockers: tuple[LearningIssue, ...] = ()
) -> P38ObservabilityAcceptanceReceipt:
    expected = set(_SECTION_CODES)
    if set(values) != expected:
        raise ValueError("acceptance section values are incomplete")
    return P38ObservabilityAcceptanceReceipt(
        schema_version=SCHEMA,
        status="pass" if all(values.values()) and not blockers else "fail",
        blockers=blockers,
        metadata={"gate": "P3.8D", "section_count": len(values)},
        **values,
    )


def _audit_metrics() -> bool:
    try:
        MetricRecord("loss", float("nan"), 0)
    except ValueError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    series = MetricSeries("loss", MetricRetentionPolicy(max_records=2))
    for step, value in enumerate((3.0, 2.0, 1.0)):
        series.add(MetricRecord("loss", value, step))
    summary = series.summary()
    return (
        invalid_rejected
        and [record.value for record in series.records] == [2.0, 1.0]
        and (summary.count, summary.last, summary.mean, summary.total)
        == (3, 1.0, 2.0, 6.0)
        and canonical_telemetry_json(summary.to_dict())
        == canonical_telemetry_json(summary.to_dict())
        and all("tome" not in name for name in ("loss", "gradient_norm"))
    )


def _audit_hooks() -> bool:
    context = HookContext("audit", "loop_start", 1, 0)
    ordered = dispatch_hooks(
        (_Hook("z", priority=1), _Hook("a", priority=0)), HookPolicy(), context
    )
    try:
        context.run_id = "mutated"
    except (AttributeError, TypeError):
        immutable = True
    else:
        immutable = False
    try:
        dispatch_hooks((_Hook("duplicate"), _Hook("duplicate")), HookPolicy(), context)
    except ValueError:
        duplicates_rejected = True
    else:
        duplicates_rejected = False
    fail_fast = dispatch_hooks(
        (_Hook("fail", result=HookResult("fail")), _Hook("later")),
        HookPolicy(),
        context,
    )
    continued = dispatch_hooks(
        (_Hook("fail", result=HookResult("fail")),),
        HookPolicy("warn_and_continue"),
        context,
    )
    disabled = dispatch_hooks(
        (_Hook("disable", result=HookResult("fail")),),
        HookPolicy("disable_hook"),
        context,
    )
    unsupported = dispatch_hooks(
        (_Hook("other", supported_events=("step_end",)),), HookPolicy(), context
    )
    skipped = dispatch_hooks((_Hook("skip"),), HookPolicy(), context, ("skip",))
    return (
        immutable
        and [receipt.hook_id for receipt in ordered.receipts] == ["a", "z"]
        and duplicates_rejected
        and not unsupported.receipts
        and not skipped.receipts
        and len(fail_fast.receipts) == 1
        and continued.status == "warning"
        and "learning_hook_failed_continue"
        in [issue.code for issue in disabled.warnings]
        and disabled.disabled_hook_ids == ("disable",)
        and not set(HookContext.__dataclass_fields__) & _FORBIDDEN_SURFACES
    )


def _audit_loop_integration() -> bool:
    recorder = _RecordingHook()
    normal = _run_loop(hooks=(recorder,), checkpoint=lambda execution: "receipt")
    learning_failure = _run_loop(objective=_RaisingObjective())
    checkpoint_failure = _run_loop(checkpoint=lambda execution: _raise())
    start_source = _CountingBatchSource((_batch(),))
    start_failure = _run_loop(
        hooks=(_FailingEventHook("loop_start"),), batch_source=start_source
    )
    step_start = _run_loop(hooks=(_FailingEventHook("step_start"),))
    checkpoint_calls: list[object] = []
    step_end = _run_loop(
        hooks=(_FailingEventHook("step_end"),),
        checkpoint=lambda execution: checkpoint_calls.append(execution) or "receipt",
    )
    checkpoint_stop = _run_loop(
        hooks=(_FailingEventHook("checkpoint"),),
        checkpoint=lambda execution: "receipt",
        max_steps=2,
    )
    terminal = _run_loop(hooks=(_FailingEventHook("loop_end"),))
    disabled_hook = _FailingEventHook("batch_received", hook_id="disabled")
    disabled = _run_loop(
        hooks=(disabled_hook,), hook_policy=HookPolicy("disable_hook"), max_steps=2
    )
    observed = [event[0] for event in recorder.events]
    sequences = [event[1] for event in recorder.events]
    return (
        observed
        == [
            "loop_start",
            "batch_received",
            "step_start",
            "step_end",
            "checkpoint",
            "loop_end",
        ]
        and sequences == list(range(1, len(sequences) + 1))
        and normal.checkpoints == ("receipt",)
        and learning_failure.stop_reason == "learning_step_failure"
        and learning_failure.hook_events[-1] == "failure"
        and "step_end" not in learning_failure.hook_events
        and checkpoint_failure.stop_reason == "checkpoint_failure"
        and checkpoint_failure.hook_events[-1] == "failure"
        and not checkpoint_failure.checkpoints
        and "checkpoint" not in checkpoint_failure.hook_events
        and start_failure.batches_consumed == 0
        and start_source.next_calls == 0
        and step_start.steps_completed == 0
        and step_end.steps_completed == 1
        and not checkpoint_calls
        and checkpoint_stop.steps_completed == 1
        and terminal.stop_reason == "hook_failure"
        and disabled.status == "pass"
        and disabled_hook.calls == 2
    )


def _audit_run_reporting() -> bool:
    plain = _run_loop()
    reported = _run_loop(emit_run_report=True)
    resumed = _run_loop(emit_run_report=True, starting_global_step=40, max_steps=2)
    report = reported.report
    return (
        plain.report is None
        and report is not None
        and report.schema_version == "radjax.learning_run_report.v1"
        and report.status.global_step == reported.global_step
        and resumed.report.status.steps_completed == 2
        and resumed.report.status.global_step == 42
        and [metric.name for metric in report.metrics]
        == sorted(metric.name for metric in report.metrics)
        and report.lifecycle.events == reported.hook_events
        and report.to_dict() == report.to_dict()
        and report.to_json() == report.to_json()
        and report.to_dict()["metric_summary_source"] == "bounded_history"
        and "parameters" not in report.to_json()
        and plain.final_execution.parameters == reported.final_execution.parameters
    )


def _audit_deterministic_replay() -> bool:
    def replay():
        return _run_loop(emit_run_report=True, checkpoint=lambda execution: "receipt")

    first, second = replay(), replay()
    return (
        first.status,
        first.stop_reason,
        first.steps_completed,
        first.global_step,
        first.batches_consumed,
        first.metrics,
        first.checkpoints,
        first.warnings,
        first.hook_events,
        first.hook_blockers,
        first.report.to_dict(),
        first.report.to_json(),
    ) == (
        second.status,
        second.stop_reason,
        second.steps_completed,
        second.global_step,
        second.batches_consumed,
        second.metrics,
        second.checkpoints,
        second.warnings,
        second.hook_events,
        second.hook_blockers,
        second.report.to_dict(),
        second.report.to_json(),
    )


def _audit_failure_paths() -> bool:
    hook = _run_loop(emit_run_report=True, hooks=(_FailingEventHook("loop_start"),))
    learning = _run_loop(
        emit_run_report=True,
        objective=_RaisingObjective(),
        hooks=(_FailingEventHook("failure"),),
    )
    checkpoint = _run_loop(
        emit_run_report=True,
        checkpoint=lambda execution: _raise(),
        hooks=(_FailingEventHook("failure"),),
    )
    return (
        hook.stop_reason == "hook_failure"
        and hook.hook_blockers
        and learning.stop_reason == "learning_step_failure"
        and learning.hook_blockers
        and checkpoint.stop_reason == "checkpoint_failure"
        and not checkpoint.checkpoints
        and all(
            result.report.status.status == result.status
            for result in (hook, learning, checkpoint)
        )
        and all(
            result.report.issues.hook_blocker_codes
            == tuple(issue.code for issue in result.hook_blockers)
            for result in (hook, learning, checkpoint)
        )
    )


def _audit_observer_only_boundary() -> bool:
    plain, reported = _run_loop(), _run_loop(emit_run_report=True)
    return (
        not set(HookContext.__dataclass_fields__) & _FORBIDDEN_SURFACES
        and plain.final_execution.parameters == reported.final_execution.parameters
        and plain.final_execution.optimizer_state
        == reported.final_execution.optimizer_state
        and plain.final_execution.learning_state
        == reported.final_execution.learning_state
        and (plain.stop_reason, plain.steps_completed)
        == (reported.stop_reason, reported.steps_completed)
    )


def _audit_bounded_history() -> bool:
    result = _run_loop(emit_run_report=True, max_steps=3, metric_history_limit=2)
    report = result.report
    summary_count = sum(metric.count for metric in report.metrics)
    return (
        len(result.metrics) == 2
        and summary_count == len(result.metrics)
        and report.to_dict()["metric_summary_source"] == "bounded_history"
        and "complete_history" not in report.to_json()
    )


def _audit_import_boundary() -> bool:
    forbidden = (
        "torch",
        "transformers",
        "tensorflow",
        "wandb",
        "mlflow",
        "tensorboard",
        "requests",
        "httpx",
        "jax",
        "optax",
        "radjax_tome",
    )
    sources = (
        _repo_root() / "src" / "radjax_student" / "learning" / "hooks.py",
        _repo_root() / "src" / "radjax_student" / "learning" / "run_report.py",
        Path(__file__),
    )
    return all(
        f"import {name}" not in source.read_text(encoding="utf-8")
        and f"from {name}" not in source.read_text(encoding="utf-8")
        for source in sources
        for name in forbidden
    )


def _audit_documentation() -> bool:
    root = _repo_root()
    return all(
        (root / path).is_file()
        for path in (
            "docs/P3_8A_HOOK_LIFECYCLE_AND_FAILURE_POLICY.md",
            "docs/P3_8B_LEARNING_LOOP_HOOK_INTEGRATION.md",
            "docs/P3_8C_DETERMINISTIC_RUN_REPORTING.md",
            "docs/P3_8D_OBSERVABILITY_GOLDEN_ACCEPTANCE_GATE.md",
            "docs/P3_8_METRICS_HOOKS_AND_REPORTING.md",
        )
    )


def _audit_test_inventory() -> bool:
    root = _repo_root()
    paths = tuple(
        root / path
        for path in (
            "tests/test_hooks.py",
            "tests/test_learning_loop_hooks.py",
            "tests/test_learning_run_report.py",
            "tests/test_p3_8_observability_acceptance.py",
        )
    )
    forbidden = ("assert True", "pytest.skip", "\n    pass\n")
    return all(
        path.is_file()
        and not any(token in path.read_text(encoding="utf-8") for token in forbidden)
        for path in paths
    )


def _run_loop(
    *,
    emit_run_report: bool = False,
    hooks=(),
    hook_policy: HookPolicy | None = None,
    checkpoint=None,
    objective=None,
    max_steps: int = 1,
    metric_history_limit: int = 64,
    starting_global_step: int = 0,
    batch_source=None,
):
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

    architecture_module = importlib.import_module("radjax_student" + ".architecture")
    testing_module = importlib.import_module("radjax_student" + ".architecture.testing")
    architecture = testing_module.FakeArchitecturePlugin()
    optimizer = SgdOptimizer()
    optimizer_config = OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1)
    learning_state = LearningState("p3-8-acceptance", global_step=starting_global_step)
    catalog = architecture.describe_parameters()
    optimizer_state = optimizer.initialize_state(
        OptimizerInitRequest(
            optimizer_config,
            catalog,
            architecture.resolve_update_scope(
                learning_state.active_update_scope, catalog
            ),
        )
    ).optimizer_state
    return run_learning_loop(
        config=LearningLoopConfig(
            max_steps=max_steps,
            metric_history_limit=metric_history_limit,
            checkpoint_every_n_steps=1 if checkpoint is not None else None,
        ),
        architecture=architecture,
        architecture_config=architecture_module.ArchitectureConfig(
            architecture.architecture_id, sequence_length=4
        ),
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        learning_state=learning_state,
        parameters={"head.weight": 0.0, "trunk.bias": 0.0, "trunk.weight": 0.0},
        objective=_LinearObjective() if objective is None else objective,
        batch_source=SyntheticBatchSource((_batch(),) * 4)
        if batch_source is None
        else batch_source,
        checkpoint=checkpoint,
        hooks=hooks,
        hook_policy=HookPolicy() if hook_policy is None else hook_policy,
        emit_run_report=emit_run_report,
    )


def _batch() -> LearningBatch:
    return LearningBatch(
        "p3-8-batch",
        inputs={"token_ids": {"rank": 2, "sequence_length": 1, "x": 1.0}},
        targets={"target": {"y": 3.0}},
    )


@dataclass(frozen=True)
class _LinearObjective:
    def evaluate(self, parameters, batch):
        error = (
            parameters["trunk.weight"] * float(batch.inputs["token_ids"]["x"])
            + parameters["head.weight"]
            - float(batch.targets["target"]["y"])
        )
        return error * error, {
            "head.weight": 2 * error,
            "trunk.bias": 0.0,
            "trunk.weight": 2 * error,
        }


@dataclass(frozen=True)
class _RaisingObjective:
    def evaluate(self, parameters, batch):
        del parameters, batch
        raise RuntimeError("acceptance failure")


@dataclass(frozen=True)
class _Hook:
    hook_id: str
    priority: int = 0
    result: HookResult = HookResult()
    supported_events: tuple[str, ...] = ("loop_start",)

    def on_event(self, context):
        del context
        return self.result


@dataclass
class _RecordingHook:
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
    events: list[tuple[str, int]] = field(default_factory=list)

    def on_event(self, context):
        self.events.append((context.event_type, context.event_sequence))
        return HookResult()


@dataclass
class _FailingEventHook:
    event: str
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
    calls: int = 0

    def on_event(self, context):
        self.calls += 1
        return HookResult("fail") if context.event_type == self.event else HookResult()


@dataclass
class _CountingBatchSource:
    batches: tuple[LearningBatch, ...]
    source_id: str = "p3-8-acceptance"
    position: int = 0
    next_calls: int = 0

    def next_batch(self):
        self.next_calls += 1
        if self.position >= len(self.batches):
            return None
        batch = self.batches[self.position]
        self.position += 1
        return batch

    def state_dict(self):
        return {"source_id": self.source_id, "position": self.position}

    def load_state_dict(self, state):
        if state["source_id"] != self.source_id:
            raise ValueError("batch source state mismatch")
        self.position = int(state["position"])


def _raise():
    raise RuntimeError("acceptance failure")


def _receipt_flags(receipt: P38ObservabilityAcceptanceReceipt) -> dict[str, bool]:
    return {name: getattr(receipt, name) for name in _SECTION_CODES}


def _validate_safe_metadata(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("acceptance metadata must be a mapping")
    for key, item in value.items():
        if key in _FORBIDDEN_METADATA:
            raise ValueError(f"acceptance metadata cannot contain {key}")
        if isinstance(item, Mapping):
            _validate_safe_metadata(item)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_FORBIDDEN_SURFACES = {
    "parameters",
    "gradients",
    "optimizer_state",
    "architecture_state",
    "runtime_state",
    "checkpoint_payload",
    "raw_batch",
    "update_scope",
    "objective_scope",
}


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the P3.8 observability acceptance gate"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_p3_8_observability_acceptance()
    if args.json:
        print(receipt.to_json())
    else:
        labels = (
            ("metrics contract", receipt.metrics_contract_valid),
            ("hook contract", receipt.hook_contract_valid),
            ("loop integration", receipt.loop_integration_valid),
            ("run reporting", receipt.run_reporting_valid),
            ("deterministic replay", receipt.deterministic_replay_valid),
            ("failure paths", receipt.failure_paths_valid),
            ("observer-only boundary", receipt.observer_only_boundary_valid),
            ("bounded-history honesty", receipt.bounded_history_claim_valid),
            ("import boundary", receipt.import_boundary_valid),
            ("documentation", receipt.documentation_valid),
            ("test inventory", receipt.test_inventory_valid),
        )
        print("P3.8 Observability Acceptance")
        print(f"  status: {receipt.status.upper()}")
        for label, valid in labels:
            print(f"  {label}: {'pass' if valid else 'fail'}")
        print(f"  blockers: {len(receipt.blockers)}")
        print(f"  warnings: {len(receipt.warnings)}")
        print("  note: no model-quality, Tome, evaluation, or telemetry claim")
    return 0 if receipt.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
