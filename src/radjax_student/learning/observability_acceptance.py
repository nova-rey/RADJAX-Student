"""Deterministic in-memory P3.8 observability acceptance audit."""

from __future__ import annotations

import argparse
import ast
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
class P38AuditDependencies:
    metric_series_factory: Callable[..., Any]
    dispatch_hooks_fn: Callable[..., Any]
    run_loop_fn: Callable[..., Any]
    build_report_fn: Callable[..., Any]
    source_loader: Callable[[Path], str]
    path_exists_fn: Callable[[Path], bool]


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


def run_p3_8_observability_acceptance(
    dependencies: P38AuditDependencies | None = None,
) -> P38ObservabilityAcceptanceReceipt:
    """Audit the completed P3.8 stack without changing learning behavior."""

    deps = _default_dependencies() if dependencies is None else dependencies
    audits: tuple[tuple[str, Callable[[P38AuditDependencies], bool]], ...] = (
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
            values[field_name] = bool(audit(deps))
        except Exception as exc:
            values[field_name] = False
            blockers.append(
                LearningIssue(
                    "p3_8_internal_error",
                    "P3.8 acceptance audit raised unexpectedly",
                    {"section": field_name, "exception_type": type(exc).__name__},
                )
            )
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


def _audit_metrics(deps: P38AuditDependencies) -> bool:
    try:
        MetricRecord("loss", float("nan"), 0)
    except ValueError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    series = deps.metric_series_factory("loss", MetricRetentionPolicy(max_records=2))
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


def _audit_hooks(deps: P38AuditDependencies) -> bool:
    context = HookContext("audit", "loop_start", 1, 0)
    ordered = deps.dispatch_hooks_fn(
        (_Hook("z", priority=1), _Hook("a", priority=0)), HookPolicy(), context
    )
    try:
        context.run_id = "mutated"
    except (AttributeError, TypeError):
        immutable = True
    else:
        immutable = False
    try:
        deps.dispatch_hooks_fn(
            (_Hook("duplicate"), _Hook("duplicate")), HookPolicy(), context
        )
    except ValueError:
        duplicates_rejected = True
    else:
        duplicates_rejected = False
    fail_fast = deps.dispatch_hooks_fn(
        (_Hook("fail", result=HookResult("fail")), _Hook("later")),
        HookPolicy(),
        context,
    )
    continued = deps.dispatch_hooks_fn(
        (_Hook("fail", result=HookResult("fail")),),
        HookPolicy("warn_and_continue"),
        context,
    )
    disabled = deps.dispatch_hooks_fn(
        (_Hook("disable", result=HookResult("fail")),),
        HookPolicy("disable_hook"),
        context,
    )
    unsupported = deps.dispatch_hooks_fn(
        (_Hook("other", supported_events=("step_end",)),), HookPolicy(), context
    )
    skipped = deps.dispatch_hooks_fn((_Hook("skip"),), HookPolicy(), context, ("skip",))
    invalid = deps.dispatch_hooks_fn(
        (_Hook("invalid", result=object()),), HookPolicy(), context
    )
    raised = deps.dispatch_hooks_fn((_RaisingHook(),), HookPolicy(), context)
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
        and "learning_hook_result_invalid" in [issue.code for issue in invalid.blockers]
        and invalid.receipts[0].metadata["returned_type"] == "object"
        and "learning_hook_failed" in [issue.code for issue in raised.blockers]
        and raised.receipts[0].metadata["exception_type"] == "RuntimeError"
        and not set(HookContext.__dataclass_fields__) & _FORBIDDEN_SURFACES
    )


def _audit_loop_integration(deps: P38AuditDependencies) -> bool:
    recorder = _RecordingHook()
    normal = deps.run_loop_fn(hooks=(recorder,), checkpoint=lambda execution: "receipt")
    learning_failure = deps.run_loop_fn(objective=_RaisingObjective())
    checkpoint_failure = deps.run_loop_fn(checkpoint=lambda execution: _raise())
    start_source = _CountingBatchSource((_batch(),))
    start_failure = deps.run_loop_fn(
        hooks=(_FailingEventHook("loop_start"),), batch_source=start_source
    )
    step_start = deps.run_loop_fn(hooks=(_FailingEventHook("step_start"),))
    checkpoint_calls: list[object] = []
    step_end = deps.run_loop_fn(
        hooks=(_FailingEventHook("step_end"),),
        checkpoint=lambda execution: checkpoint_calls.append(execution) or "receipt",
    )
    checkpoint_stop = deps.run_loop_fn(
        hooks=(_FailingEventHook("checkpoint"),),
        checkpoint=lambda execution: "receipt",
        max_steps=2,
    )
    terminal = deps.run_loop_fn(hooks=(_FailingEventHook("loop_end"),))
    disabled_hook = _FailingEventHook("batch_received", hook_id="disabled")
    disabled = deps.run_loop_fn(
        hooks=(disabled_hook,), hook_policy=HookPolicy("disable_hook"), max_steps=2
    )
    flow = deps.run_loop_fn(
        hooks=(
            _Hook(
                "flow",
                result=HookResult(
                    "warning",
                    metrics=(MetricRecord("hook.metric", 1, 0),),
                    warnings=(LearningIssue("hook.warning", "flow"),),
                ),
            ),
        )
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
        and normal.hook_events == tuple(observed)
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
        and [issue.code for issue in start_failure.hook_blockers]
        == ["learning_hook_failed"]
        and step_start.steps_completed == 0
        and step_end.steps_completed == 1
        and not checkpoint_calls
        and checkpoint_stop.steps_completed == 1
        and terminal.stop_reason == "hook_failure"
        and disabled.status == "pass"
        and disabled_hook.calls == 2
        and "hook.metric" in [metric.name for metric in flow.metrics]
        and "hook.warning" in [warning.code for warning in flow.warnings]
    )


def _audit_run_reporting(deps: P38AuditDependencies) -> bool:
    plain = deps.run_loop_fn()
    reported = deps.run_loop_fn(emit_run_report=True)
    resumed = deps.run_loop_fn(
        emit_run_report=True, starting_global_step=40, max_steps=2
    )
    report = reported.report
    rebuilt = deps.build_report_fn(
        loop_result=plain,
        run_id="p3-8-acceptance",
        update_scope="whole_student",
        objective_scope="final_output",
    )
    rebuilt_resumed = deps.build_report_fn(
        loop_result=resumed,
        run_id="p3-8-acceptance",
        update_scope="whole_student",
        objective_scope="final_output",
    )
    rebuilt_again = deps.build_report_fn(
        loop_result=plain,
        run_id="p3-8-acceptance",
        update_scope="whole_student",
        objective_scope="final_output",
    )
    blocked = deps.run_loop_fn(
        hooks=(_FailingEventHook("loop_start"),), emit_run_report=True
    )
    blocked_report = deps.build_report_fn(
        loop_result=blocked,
        run_id="p3-8-acceptance",
        update_scope="whole_student",
        objective_scope="final_output",
    )
    return (
        plain.report is None
        and report is not None
        and report.schema_version == "radjax.learning_run_report.v1"
        and report.status.global_step == reported.global_step
        and rebuilt.status.global_step == plain.global_step
        and rebuilt_resumed.status.global_step == resumed.global_step
        and resumed.report.status.steps_completed == 2
        and resumed.report.status.global_step == 42
        and [metric.name for metric in report.metrics]
        == sorted(metric.name for metric in report.metrics)
        and report.lifecycle.events == reported.hook_events
        and report.to_dict() == report.to_dict()
        and report.to_json() == report.to_json()
        and rebuilt.to_json() == rebuilt.to_json()
        and rebuilt.to_json() == rebuilt_again.to_json()
        and tuple(blocked_report.issues.hook_blocker_codes)
        == tuple(issue.code for issue in blocked.hook_blockers)
        and not set(blocked_report.issues.warning_codes)
        & set(blocked_report.issues.hook_blocker_codes)
        and report.to_dict()["metric_summary_source"] == "bounded_history"
        and "parameters" not in report.to_json()
        and plain.final_execution.parameters == reported.final_execution.parameters
    )


def _audit_deterministic_replay(deps: P38AuditDependencies) -> bool:
    def replay():
        return deps.run_loop_fn(
            emit_run_report=True, checkpoint=lambda execution: "receipt"
        )

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


def _audit_failure_paths(deps: P38AuditDependencies) -> bool:
    hook = deps.run_loop_fn(
        emit_run_report=True, hooks=(_FailingEventHook("loop_start"),)
    )
    learning = deps.run_loop_fn(
        emit_run_report=True,
        objective=_RaisingObjective(),
        hooks=(_FailingEventHook("failure"),),
    )
    checkpoint = deps.run_loop_fn(
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


def _audit_observer_only_boundary(deps: P38AuditDependencies) -> bool:
    plain, reported = deps.run_loop_fn(), deps.run_loop_fn(emit_run_report=True)
    metadata = {"sentinel": "unchanged"}
    deps.build_report_fn(
        loop_result=plain,
        run_id="p3-8-acceptance",
        update_scope="whole_student",
        objective_scope="final_output",
        metadata=metadata,
    )
    return (
        not set(HookContext.__dataclass_fields__) & _FORBIDDEN_SURFACES
        and plain.final_execution.parameters == reported.final_execution.parameters
        and plain.final_execution.optimizer_state
        == reported.final_execution.optimizer_state
        and plain.final_execution.learning_state
        == reported.final_execution.learning_state
        and (plain.stop_reason, plain.steps_completed)
        == (reported.stop_reason, reported.steps_completed)
        and metadata == {"sentinel": "unchanged"}
    )


def _audit_bounded_history(deps: P38AuditDependencies) -> bool:
    result = deps.run_loop_fn(emit_run_report=True, max_steps=3, metric_history_limit=2)
    report = deps.build_report_fn(
        loop_result=result,
        run_id="p3-8-acceptance",
        update_scope="whole_student",
        objective_scope="final_output",
    )
    summary_count = sum(metric.count for metric in report.metrics)
    return (
        len(result.metrics) == 2
        and summary_count == len(result.metrics)
        and report.to_dict()["metric_summary_source"] == "bounded_history"
        and "complete_history" not in report.to_json()
    )


def _audit_import_boundary(deps: P38AuditDependencies) -> bool:
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
    return not any(
        _source_has_forbidden_import(deps.source_loader(source), forbidden)
        for source in sources
    )


def _audit_documentation(deps: P38AuditDependencies) -> bool:
    root = _repo_root()
    return all(
        deps.path_exists_fn(root / path)
        for path in (
            "docs/P3_8A_HOOK_LIFECYCLE_AND_FAILURE_POLICY.md",
            "docs/P3_8B_LEARNING_LOOP_HOOK_INTEGRATION.md",
            "docs/P3_8C_DETERMINISTIC_RUN_REPORTING.md",
            "docs/P3_8D_OBSERVABILITY_GOLDEN_ACCEPTANCE_GATE.md",
            "docs/P3_8_METRICS_HOOKS_AND_REPORTING.md",
        )
    )


def _audit_test_inventory(deps: P38AuditDependencies) -> bool:
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
    return all(
        deps.path_exists_fn(path)
        and not _has_test_placeholder(deps.source_loader(path))
        for path in paths
    )


def _default_dependencies() -> P38AuditDependencies:
    from radjax_student.learning.run_report import build_learning_run_report

    return P38AuditDependencies(
        metric_series_factory=MetricSeries,
        dispatch_hooks_fn=dispatch_hooks,
        run_loop_fn=_run_loop,
        build_report_fn=build_learning_run_report,
        source_loader=lambda path: path.read_text(encoding="utf-8"),
        path_exists_fn=lambda path: path.is_file(),
    )


def _source_has_forbidden_import(source: str, forbidden_roots: tuple[str, ...]) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in forbidden_roots for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in forbidden_roots:
                return True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
        ):
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                if node.args[0].value.split(".")[0] in forbidden_roots:
                    return True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                if node.args[0].value.split(".")[0] in forbidden_roots:
                    return True
    return False


def _has_test_placeholder(source: str) -> bool:
    tree = ast.parse(source)
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        if not function.name.startswith("test_"):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Pass):
                return True
            if (
                isinstance(node, ast.Assert)
                and isinstance(node.test, ast.Constant)
                and node.test.value is True
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.startswith("test_")
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pytest"
                and node.func.attr == "skip"
            ):
                return True
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("tests")
            ):
                if any(alias.name.startswith("test_") for alias in node.names):
                    return True
    return False


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


@dataclass(frozen=True)
class _RaisingHook:
    hook_id: str = "raise"
    priority: int = 0
    supported_events: tuple[str, ...] = ("loop_start",)

    def on_event(self, context):
        del context
        raise RuntimeError("acceptance hook failure")


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


def main(
    argv: tuple[str, ...] | None = None,
    dependencies: P38AuditDependencies | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run the P3.8 observability acceptance gate"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_p3_8_observability_acceptance(dependencies)
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
