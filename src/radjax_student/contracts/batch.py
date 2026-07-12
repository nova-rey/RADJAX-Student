"""Finite-JSON batch identity shared by architecture and learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from radjax_student.contracts._json import (
    freeze_json_mapping,
    json_value,
    mapping,
    nonempty_string,
)
from radjax_student.contracts.scopes import ObjectiveScope


@dataclass(frozen=True)
class LearningBatch:
    batch_id: str
    inputs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    targets: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    weights: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    objective_scope: ObjectiveScope = ObjectiveScope()

    def __post_init__(self) -> None:
        nonempty_string(self.batch_id, "batch_id")
        for name in ("inputs", "targets", "weights", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, freeze_json_mapping(value))
        if not isinstance(self.objective_scope, ObjectiveScope):
            raise TypeError("objective_scope must be ObjectiveScope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "inputs": json_value(self.inputs),
            "targets": json_value(self.targets),
            "weights": json_value(self.weights),
            "metadata": json_value(self.metadata),
            "objective_scope": self.objective_scope.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningBatch:
        return cls(
            str(payload["batch_id"]),
            mapping(payload.get("inputs", {}), "inputs"),
            mapping(payload.get("targets", {}), "targets"),
            mapping(payload.get("weights", {}), "weights"),
            mapping(payload.get("metadata", {}), "metadata"),
            ObjectiveScope.from_dict(
                mapping(payload.get("objective_scope", {}), "objective_scope")
            ),
        )


__all__ = ["LearningBatch"]
