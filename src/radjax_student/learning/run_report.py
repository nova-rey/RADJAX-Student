"""Pure deterministic summaries of completed generic learning runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.learning._json import (
    finite_number,
    freeze_json_mapping,
    json_value,
    nonempty_string,
    nonnegative_int,
    strings,
    unique_strings,
)
from radjax_student.learning.errors import LearningIssue
from radjax_student.learning.telemetry import MetricSeries

SCHEMA = "radjax.learning_run_report.v1"
CLAIMS = (
    "generic_learning_run_report_generated",
    "deterministic_serialization",
    "bounded_metric_reporting",
    "observer_only_hook_reporting",
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
)
EVENTS = {
    "loop_start",
    "batch_received",
    "step_start",
    "step_end",
    "checkpoint",
    "loop_end",
    "failure",
}
_FORBIDDEN_METADATA_FIELDS = {
    "parameters",
    "gradients",
    "optimizer_state",
    "architecture_state",
    "runtime_handle",
    "raw_batch",
    "traceback",
}


@dataclass(frozen=True)
class RunStatusSummary:
    status: Literal["pass", "fail"]
    stop_reason: str
    steps_completed: int
    global_step: int

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("run status must be pass or fail")
        nonempty_string(self.stop_reason, "stop_reason")
        nonnegative_int(self.steps_completed, "steps_completed")
        nonnegative_int(self.global_step, "global_step")
        if self.global_step < self.steps_completed:
            raise ValueError("global_step cannot precede steps_completed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps_completed": self.steps_completed,
            "global_step": self.global_step,
        }


@dataclass(frozen=True)
class RunMetricSummary:
    name: str
    count: int
    last: float
    minimum: float
    maximum: float
    mean: float
    sum: float

    def __post_init__(self) -> None:
        nonempty_string(self.name, "metric name")
        nonnegative_int(self.count, "metric count")
        if self.count == 0:
            raise ValueError("metric count must be positive")
        for name in ("last", "minimum", "maximum", "mean", "sum"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "last": self.last,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "sum": self.sum,
        }


@dataclass(frozen=True)
class RunLifecycleSummary:
    events: tuple[str, ...]
    event_count: int
    first_event: str | None
    last_event: str | None

    def __post_init__(self) -> None:
        events = strings(self.events, "events")
        if any(event not in EVENTS for event in events):
            raise ValueError("unsupported lifecycle event")
        nonnegative_int(self.event_count, "event_count")
        if self.event_count != len(events):
            raise ValueError("event_count must match events")
        if events:
            if self.first_event != events[0] or self.last_event != events[-1]:
                raise ValueError("lifecycle boundaries must match events")
        elif self.first_event is not None or self.last_event is not None:
            raise ValueError("empty lifecycle cannot have boundary events")
        object.__setattr__(self, "events", events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": list(self.events),
            "event_count": self.event_count,
            "first_event": self.first_event,
            "last_event": self.last_event,
        }


@dataclass(frozen=True)
class RunIssueSummary:
    warning_codes: tuple[str, ...]
    hook_blocker_codes: tuple[str, ...]
    warning_count: int | None = None
    hook_blocker_count: int | None = None

    def __post_init__(self) -> None:
        warning_codes = strings(self.warning_codes, "warning_codes")
        blocker_codes = strings(self.hook_blocker_codes, "hook_blocker_codes")
        warning_count = (
            len(warning_codes) if self.warning_count is None else self.warning_count
        )
        blocker_count = (
            len(blocker_codes)
            if self.hook_blocker_count is None
            else self.hook_blocker_count
        )
        nonnegative_int(warning_count, "warning_count")
        nonnegative_int(blocker_count, "hook_blocker_count")
        if warning_count != len(warning_codes) or blocker_count != len(blocker_codes):
            raise ValueError("issue counts must match code occurrences")
        object.__setattr__(self, "warning_codes", warning_codes)
        object.__setattr__(self, "hook_blocker_codes", blocker_codes)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "hook_blocker_count", blocker_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning_codes": list(self.warning_codes),
            "hook_blocker_codes": list(self.hook_blocker_codes),
            "warning_count": self.warning_count,
            "hook_blocker_count": self.hook_blocker_count,
        }


@dataclass(frozen=True)
class RunCheckpointSummary:
    receipts: tuple[str, ...]
    count: int | None = None

    def __post_init__(self) -> None:
        receipts = strings(self.receipts, "receipts")
        count = len(receipts) if self.count is None else self.count
        nonnegative_int(count, "checkpoint count")
        if count != len(receipts):
            raise ValueError("checkpoint count must match receipts")
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "count", count)

    def to_dict(self) -> dict[str, Any]:
        return {"receipts": list(self.receipts), "count": self.count}


@dataclass(frozen=True)
class RunScopeSummary:
    update_scope: str
    objective_scope: str

    def __post_init__(self) -> None:
        nonempty_string(self.update_scope, "update_scope")
        nonempty_string(self.objective_scope, "objective_scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_scope": self.update_scope,
            "objective_scope": self.objective_scope,
        }


@dataclass(frozen=True)
class LearningRunReport:
    run_id: str
    status: RunStatusSummary
    metrics: tuple[RunMetricSummary, ...]
    lifecycle: RunLifecycleSummary
    issues: RunIssueSummary
    checkpoints: RunCheckpointSummary
    scopes: RunScopeSummary
    hook_blockers: tuple[LearningIssue, ...] = ()
    schema_version: str = SCHEMA
    claims_made: tuple[str, ...] = CLAIMS
    claims_not_made: tuple[str, ...] = NON_CLAIMS
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonempty_string(self.run_id, "run_id")
        if self.schema_version != SCHEMA:
            raise ValueError("unsupported learning run report schema")
        if not all(
            isinstance(value, expected)
            for value, expected in (
                (self.status, RunStatusSummary),
                (self.lifecycle, RunLifecycleSummary),
                (self.issues, RunIssueSummary),
                (self.checkpoints, RunCheckpointSummary),
                (self.scopes, RunScopeSummary),
            )
        ):
            raise TypeError("learning run report nested models are invalid")
        metrics = tuple(self.metrics)
        blockers = tuple(self.hook_blockers)
        if any(not isinstance(metric, RunMetricSummary) for metric in metrics):
            raise TypeError("metrics must contain RunMetricSummary")
        if tuple(metric.name for metric in metrics) != tuple(
            sorted(metric.name for metric in metrics)
        ):
            raise ValueError("metrics must be ordered by name")
        if any(not isinstance(blocker, LearningIssue) for blocker in blockers):
            raise TypeError("hook_blockers must contain LearningIssue")
        if (
            tuple(blocker.code for blocker in blockers)
            != self.issues.hook_blocker_codes
        ):
            raise ValueError("hook blocker codes must match hook blockers")
        claims_made = unique_strings(self.claims_made, "claims_made")
        claims_not_made = unique_strings(self.claims_not_made, "claims_not_made")
        if claims_made != CLAIMS or not set(NON_CLAIMS).issubset(claims_not_made):
            raise ValueError("learning run report claims are invalid")
        _validate_metadata(self.metadata)
        for blocker in blockers:
            _validate_metadata(blocker.details)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "hook_blockers", blockers)
        object.__setattr__(self, "claims_made", claims_made)
        object.__setattr__(self, "claims_not_made", claims_not_made)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "lifecycle": self.lifecycle.to_dict(),
            "issues": self.issues.to_dict(),
            "checkpoints": self.checkpoints.to_dict(),
            "scopes": self.scopes.to_dict(),
            "hook_blockers": [blocker.to_dict() for blocker in self.hook_blockers],
            "claims_made": list(self.claims_made),
            "claims_not_made": list(self.claims_not_made),
            "metadata": json_value(self.metadata),
            "metric_summary_source": "bounded_history",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def build_learning_run_report(
    *,
    loop_result: Any,
    run_id: str,
    update_scope: str,
    objective_scope: str,
    metadata: Mapping[str, Any] | None = None,
) -> LearningRunReport:
    """Build a report from retained loop observations without changing the run."""

    from radjax_student.steps.loop import LearningLoopResult

    if not isinstance(loop_result, LearningLoopResult):
        raise TypeError("loop_result must be LearningLoopResult")
    summaries: list[RunMetricSummary] = []
    records_by_name: dict[str, list[Any]] = {}
    for record in loop_result.metrics:
        records_by_name.setdefault(record.name, []).append(record)
    for name in sorted(records_by_name):
        series = MetricSeries(name)
        for record in records_by_name[name]:
            series.add(record)
        summary = series.summary()
        summaries.append(
            RunMetricSummary(
                name=summary.name,
                count=summary.count,
                last=summary.last,
                minimum=summary.minimum,
                maximum=summary.maximum,
                mean=summary.mean,
                sum=summary.total,
            )
        )
    events = tuple(loop_result.hook_events)
    return LearningRunReport(
        run_id=run_id,
        status=RunStatusSummary(
            loop_result.status,
            loop_result.stop_reason,
            loop_result.steps_completed,
            loop_result.steps_completed,
        ),
        metrics=tuple(summaries),
        lifecycle=RunLifecycleSummary(
            events,
            len(events),
            events[0] if events else None,
            events[-1] if events else None,
        ),
        issues=RunIssueSummary(
            tuple(issue.code for issue in loop_result.warnings),
            tuple(issue.code for issue in loop_result.hook_blockers),
        ),
        checkpoints=RunCheckpointSummary(tuple(loop_result.checkpoints)),
        scopes=RunScopeSummary(update_scope, objective_scope),
        hook_blockers=tuple(loop_result.hook_blockers),
        metadata={} if metadata is None else metadata,
    )


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    for key, value in metadata.items():
        if key in _FORBIDDEN_METADATA_FIELDS:
            raise ValueError(f"metadata must not include {key}")
        if isinstance(value, Mapping):
            _validate_metadata(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    _validate_metadata(item)
