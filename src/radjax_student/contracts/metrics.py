"""Dependency-light metric primitives shared across learning and optimizers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

METRIC_AGGREGATIONS = ("last", "mean", "sum", "min", "max")


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    raise TypeError("metric metadata must be finite JSON data")


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    return value


@dataclass(frozen=True)
class MetricRecord:
    name: str
    value: float
    step: int
    unit: str = "unitless"
    aggregation: Literal["last", "mean", "sum", "min", "max"] = "last"
    scope: str = "learning"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be nonempty")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(float(self.value))
        ):
            raise ValueError("value must be finite")
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValueError("step must be nonnegative")
        if (
            not self.unit
            or not self.scope
            or self.aggregation not in METRIC_AGGREGATIONS
        ):
            raise ValueError("metric identity is invalid")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "scope": self.scope,
            "metadata": _json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetricRecord:
        return cls(
            str(payload["name"]),
            payload["value"],
            payload["step"],
            str(payload.get("unit", "unitless")),
            str(payload.get("aggregation", "last")),
            str(payload.get("scope", "learning")),
            payload.get("metadata", {}),
        )


__all__ = ["METRIC_AGGREGATIONS", "MetricRecord"]
