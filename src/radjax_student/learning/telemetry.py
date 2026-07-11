"""Deterministic bounded metrics, events, and passive hook contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from radjax_student.learning.models import MetricRecord, canonical_learning_json

METRIC_NAMES = (
    "loss",
    "gradient_norm",
    "parameter_norm",
    "learning_rate",
    "changed_parameter_count",
    "unchanged_parameter_count",
    "step_time_seconds",
)


@dataclass(frozen=True)
class MetricRetentionPolicy:
    kind: Literal["latest_only", "bounded_history", "summary_only", "disabled"] = (
        "bounded_history"
    )
    max_records: int = 256

    def __post_init__(self):
        if self.kind == "bounded_history" and self.max_records < 1:
            raise ValueError("bounded history requires positive max_records")


DEFAULT_RETENTION_POLICY = MetricRetentionPolicy()


@dataclass(frozen=True)
class MetricSummary:
    name: str
    count: int
    last: float
    mean: float
    minimum: float
    maximum: float
    total: float
    first_step: int
    last_step: int
    unit: str

    def to_dict(self):
        return self.__dict__.copy()


class MetricSeries:
    def __init__(
        self, name: str, policy: MetricRetentionPolicy = DEFAULT_RETENTION_POLICY
    ):
        self.name, self.policy, self._records, self._all = name, policy, [], []

    def add(self, record: MetricRecord):
        if record.name != self.name:
            raise ValueError("metric name mismatch")
        self._all.append(record)
        self._records.append(record)
        if self.policy.kind == "latest_only":
            self._records = self._records[-1:]
        elif self.policy.kind == "summary_only" or self.policy.kind == "disabled":
            self._records = []
        elif self.policy.kind == "bounded_history":
            self._records = self._records[-self.policy.max_records :]

    @property
    def records(self):
        return tuple(self._records)

    def summary(self):
        values = [r.value for r in self._all]
        return MetricSummary(
            self.name,
            len(values),
            values[-1],
            sum(values) / len(values),
            min(values),
            max(values),
            sum(values),
            self._all[0].step,
            self._all[-1].step,
            self._all[-1].unit,
        )


@dataclass(frozen=True)
class LearningEvent:
    event_type: str
    sequence: int
    step: int
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class HookContext:
    run_id: str
    global_step: int
    event_type: str
    metrics: tuple[MetricRecord, ...] = ()


def canonical_telemetry_json(value):
    return canonical_learning_json(value)
