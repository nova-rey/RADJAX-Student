"""Model-neutral batch metadata and objective contract models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.learning._json import (
    finite_number,
    freeze_json_mapping,
    json_value,
    mapping,
    nonempty_string,
    nonnegative_int,
    unique_strings,
)
from radjax_student.learning.errors import LearningIssue
from radjax_student.learning.models import MetricRecord
from radjax_student.learning.scopes import ObjectiveScope

WEIGHTING_POLICIES: tuple[str, ...] = ("uniform", "explicit_weights", "plugin_defined")
BATCH_OBJECTIVE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "tome_not_loaded",
    "teacher_logits_not_loaded",
    "gradient_not_computed",
    "learning_step_not_run",
    "training_loop_not_run",
)


@dataclass(frozen=True)
class BatchMetadata:
    sample_count: int
    sequence_length: int | None = None
    padding_policy: str = "unspecified"
    mask_summary: Mapping[str, Any] = MappingProxyType({})
    source: str | None = None
    claims_not_made: tuple[str, ...] = BATCH_OBJECTIVE_CLAIMS_NOT_MADE
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        nonnegative_int(self.sample_count, "sample_count")
        if self.sequence_length is not None:
            nonnegative_int(self.sequence_length, "sequence_length")
        nonempty_string(self.padding_policy, "padding_policy")
        if self.source is not None:
            nonempty_string(self.source, "source")
        object.__setattr__(self, "mask_summary", freeze_json_mapping(self.mask_summary))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "claims_not_made",
            unique_strings(self.claims_not_made, "claims_not_made"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "sequence_length": self.sequence_length,
            "padding_policy": self.padding_policy,
            "mask_summary": json_value(self.mask_summary),
            "source": self.source,
            "claims_not_made": list(self.claims_not_made),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BatchMetadata:
        return cls(
            sample_count=payload["sample_count"],
            sequence_length=payload.get("sequence_length"),
            padding_policy=str(payload.get("padding_policy", "unspecified")),
            mask_summary=mapping(payload.get("mask_summary", {}), "mask_summary"),
            source=payload.get("source"),
            claims_not_made=unique_strings(
                payload.get("claims_not_made", BATCH_OBJECTIVE_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class WeightingPolicy:
    kind: Literal["uniform", "explicit_weights", "plugin_defined"] = "uniform"
    weight_key: str | None = None
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.kind not in WEIGHTING_POLICIES:
            raise ValueError("weighting policy kind is unsupported")
        if self.weight_key is not None:
            nonempty_string(self.weight_key, "weight_key")
        if self.kind == "explicit_weights" and self.weight_key is None:
            raise ValueError("explicit_weights requires weight_key")
        if self.kind == "uniform" and self.weight_key is not None:
            raise ValueError("uniform weighting cannot carry weight_key")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "weight_key": self.weight_key,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WeightingPolicy:
        return cls(
            kind=str(payload.get("kind", "uniform")),
            weight_key=payload.get("weight_key"),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ObjectiveRequest:
    objective_id: str
    objective_scope: ObjectiveScope = ObjectiveScope()
    batch_reference: str | None = None
    required_outputs: tuple[str, ...] = ()
    weighting_policy: WeightingPolicy = WeightingPolicy()
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        nonempty_string(self.objective_id, "objective_id")
        if not isinstance(self.objective_scope, ObjectiveScope) or not isinstance(
            self.weighting_policy, WeightingPolicy
        ):
            raise TypeError(
                "objective scope and weighting policy must be contract models"
            )
        if self.batch_reference is not None:
            nonempty_string(self.batch_reference, "batch_reference")
        object.__setattr__(
            self,
            "required_outputs",
            unique_strings(self.required_outputs, "required_outputs"),
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "objective_scope": self.objective_scope.to_dict(),
            "batch_reference": self.batch_reference,
            "required_outputs": list(self.required_outputs),
            "weighting_policy": self.weighting_policy.to_dict(),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveRequest:
        return cls(
            objective_id=str(payload["objective_id"]),
            objective_scope=ObjectiveScope.from_dict(
                mapping(payload.get("objective_scope", {}), "objective_scope")
            ),
            batch_reference=payload.get("batch_reference"),
            required_outputs=unique_strings(
                payload.get("required_outputs", ()), "required_outputs"
            ),
            weighting_policy=WeightingPolicy.from_dict(
                mapping(payload.get("weighting_policy", {}), "weighting_policy")
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ObjectiveResult:
    objective_id: str
    loss: float
    components: Mapping[str, float] = MappingProxyType({})
    metrics: tuple[MetricRecord, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    claims_not_made: tuple[str, ...] = BATCH_OBJECTIVE_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        nonempty_string(self.objective_id, "objective_id")
        object.__setattr__(self, "loss", finite_number(self.loss, "loss"))
        components = {
            nonempty_string(name, "component name"): finite_number(value, "component")
            for name, value in self.components.items()
        }
        metrics, warnings = tuple(self.metrics), tuple(self.warnings)
        if any(not isinstance(item, MetricRecord) for item in metrics) or any(
            not isinstance(item, LearningIssue) for item in warnings
        ):
            raise TypeError(
                "objective result metrics and warnings must be typed contract values"
            )
        object.__setattr__(self, "components", MappingProxyType(components))
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "claims_not_made",
            unique_strings(self.claims_not_made, "claims_not_made"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "loss": self.loss,
            "components": dict(self.components),
            "metrics": [item.to_dict() for item in self.metrics],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveResult:
        return cls(
            objective_id=str(payload["objective_id"]),
            loss=payload["loss"],
            components=mapping(payload.get("components", {}), "components"),
            metrics=tuple(
                MetricRecord.from_dict(mapping(item, "metric"))
                for item in payload.get("metrics", ())
            ),
            warnings=tuple(
                LearningIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            claims_not_made=unique_strings(
                payload.get("claims_not_made", BATCH_OBJECTIVE_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


def canonical_objective_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
