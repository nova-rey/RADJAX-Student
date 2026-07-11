"""Immutable, serializable vocabulary for generic learning state and reporting."""

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
    mapping,
    nonempty_string,
    nonnegative_int,
    optional_string,
    unique_strings,
)
from radjax_student.learning.errors import LearningIssue
from radjax_student.learning.scopes import ObjectiveScope, UpdateScope

LEARNING_STATE_SCHEMA_VERSION = "learning_state.v1"
CHECKPOINT_POLICY_MODES: tuple[str, ...] = (
    "disabled",
    "every_n_steps",
    "on_improvement",
    "manual",
)
METRIC_AGGREGATIONS: tuple[str, ...] = ("last", "mean", "sum", "min", "max")
LEARNING_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "architecture_plugin_not_invoked",
    "gradient_not_computed",
    "optimizer_not_invoked",
    "parameter_tree_not_updated",
    "checkpoint_file_not_written",
    "training_loop_not_run",
)


@dataclass(frozen=True)
class CheckpointPolicy:
    mode: Literal["disabled", "every_n_steps", "on_improvement", "manual"] = "disabled"
    every_n_steps: int | None = None
    monitor_metric: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.mode not in CHECKPOINT_POLICY_MODES:
            raise ValueError("checkpoint policy mode is unsupported")
        if self.every_n_steps is not None:
            nonnegative_int(self.every_n_steps, "every_n_steps")
            if self.every_n_steps == 0:
                raise ValueError("every_n_steps must be positive when specified")
        monitor = optional_string(self.monitor_metric, "monitor_metric")
        if self.mode == "every_n_steps" and self.every_n_steps is None:
            raise ValueError("every_n_steps checkpoint policy requires every_n_steps")
        if self.mode == "on_improvement" and monitor is None:
            raise ValueError("on_improvement checkpoint policy requires monitor_metric")
        if self.mode in ("disabled", "manual") and (
            self.every_n_steps is not None or monitor is not None
        ):
            raise ValueError(
                "disabled or manual checkpoint policy cannot carry schedule fields"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("checkpoint policy metadata must be a mapping")
        object.__setattr__(self, "monitor_metric", monitor)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "every_n_steps": self.every_n_steps,
            "monitor_metric": self.monitor_metric,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointPolicy:
        return cls(
            mode=str(payload.get("mode", "disabled")),
            every_n_steps=payload.get("every_n_steps"),
            monitor_metric=payload.get("monitor_metric"),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class LearningConfig:
    max_steps: int = 1
    gradient_accumulation_steps: int = 1
    update_scope: UpdateScope = UpdateScope()
    objective_scope: ObjectiveScope = ObjectiveScope()
    checkpoint_policy: CheckpointPolicy = CheckpointPolicy()
    metric_policy: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    seed_reference: int | str | None = None
    debug: bool = False
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        nonnegative_int(self.max_steps, "max_steps")
        nonnegative_int(self.gradient_accumulation_steps, "gradient_accumulation_steps")
        if self.max_steps == 0 or self.gradient_accumulation_steps == 0:
            raise ValueError(
                "max_steps and gradient_accumulation_steps must be positive"
            )
        if not isinstance(self.update_scope, UpdateScope):
            raise TypeError("update_scope must be UpdateScope")
        if not isinstance(self.objective_scope, ObjectiveScope):
            raise TypeError("objective_scope must be ObjectiveScope")
        if not isinstance(self.checkpoint_policy, CheckpointPolicy):
            raise TypeError("checkpoint_policy must be CheckpointPolicy")
        if self.seed_reference is not None and (
            not isinstance(self.seed_reference, (str, int))
            or isinstance(self.seed_reference, bool)
            or (isinstance(self.seed_reference, int) and self.seed_reference < 0)
        ):
            raise ValueError(
                "seed_reference must be a nonnegative integer, string, or None"
            )
        if not isinstance(self.debug, bool):
            raise TypeError("debug must be a boolean")
        if not isinstance(self.metric_policy, Mapping) or not isinstance(
            self.metadata, Mapping
        ):
            raise TypeError("learning config policies and metadata must be mappings")
        object.__setattr__(
            self, "metric_policy", freeze_json_mapping(self.metric_policy)
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "update_scope": self.update_scope.to_dict(),
            "objective_scope": self.objective_scope.to_dict(),
            "checkpoint_policy": self.checkpoint_policy.to_dict(),
            "metric_policy": json_value(self.metric_policy),
            "seed_reference": self.seed_reference,
            "debug": self.debug,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningConfig:
        return cls(
            max_steps=payload.get("max_steps", 1),
            gradient_accumulation_steps=payload.get("gradient_accumulation_steps", 1),
            update_scope=UpdateScope.from_dict(
                mapping(payload.get("update_scope", {}), "update_scope")
            ),
            objective_scope=ObjectiveScope.from_dict(
                mapping(payload.get("objective_scope", {}), "objective_scope")
            ),
            checkpoint_policy=CheckpointPolicy.from_dict(
                mapping(payload.get("checkpoint_policy", {}), "checkpoint_policy")
            ),
            metric_policy=mapping(payload.get("metric_policy", {}), "metric_policy"),
            seed_reference=payload.get("seed_reference"),
            debug=payload.get("debug", False),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class LearningState:
    run_id: str
    global_step: int = 0
    micro_step: int = 0
    epoch: int = 0
    optimizer_step: int = 0
    runtime_state_reference: str | None = None
    active_update_scope: UpdateScope = UpdateScope()
    active_objective_scope: ObjectiveScope = ObjectiveScope()
    schema_version: str = LEARNING_STATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    claims_not_made: tuple[str, ...] = LEARNING_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        nonempty_string(self.run_id, "run_id")
        if self.schema_version != LEARNING_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported learning state schema version")
        for name in ("global_step", "micro_step", "epoch", "optimizer_step"):
            nonnegative_int(getattr(self, name), name)
        runtime_reference = optional_string(
            self.runtime_state_reference, "runtime_state_reference"
        )
        if not isinstance(self.active_update_scope, UpdateScope):
            raise TypeError("active_update_scope must be UpdateScope")
        if not isinstance(self.active_objective_scope, ObjectiveScope):
            raise TypeError("active_objective_scope must be ObjectiveScope")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("learning state metadata must be a mapping")
        object.__setattr__(self, "runtime_state_reference", runtime_reference)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        object.__setattr__(
            self,
            "claims_not_made",
            unique_strings(self.claims_not_made, "claims_not_made"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": self.epoch,
            "optimizer_step": self.optimizer_step,
            "runtime_state_reference": self.runtime_state_reference,
            "active_update_scope": self.active_update_scope.to_dict(),
            "active_objective_scope": self.active_objective_scope.to_dict(),
            "metadata": json_value(self.metadata),
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningState:
        return cls(
            run_id=str(payload["run_id"]),
            global_step=payload.get("global_step", 0),
            micro_step=payload.get("micro_step", 0),
            epoch=payload.get("epoch", 0),
            optimizer_step=payload.get("optimizer_step", 0),
            runtime_state_reference=payload.get("runtime_state_reference"),
            active_update_scope=UpdateScope.from_dict(
                mapping(payload.get("active_update_scope", {}), "active_update_scope")
            ),
            active_objective_scope=ObjectiveScope.from_dict(
                mapping(
                    payload.get("active_objective_scope", {}),
                    "active_objective_scope",
                )
            ),
            schema_version=str(
                payload.get("schema_version", LEARNING_STATE_SCHEMA_VERSION)
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
            claims_not_made=unique_strings(
                payload.get("claims_not_made", LEARNING_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


@dataclass(frozen=True)
class LearningBatch:
    """Generic finite-JSON batch description; P3.1 does not carry array objects."""

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
            batch_id=str(payload["batch_id"]),
            inputs=mapping(payload.get("inputs", {}), "inputs"),
            targets=mapping(payload.get("targets", {}), "targets"),
            weights=mapping(payload.get("weights", {}), "weights"),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
            objective_scope=ObjectiveScope.from_dict(
                mapping(payload.get("objective_scope", {}), "objective_scope")
            ),
        )


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
        nonempty_string(self.name, "name")
        object.__setattr__(self, "value", finite_number(self.value, "value"))
        nonnegative_int(self.step, "step")
        nonempty_string(self.unit, "unit")
        nonempty_string(self.scope, "scope")
        if self.aggregation not in METRIC_AGGREGATIONS:
            raise ValueError("metric aggregation is unsupported")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metric metadata must be a mapping")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "scope": self.scope,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetricRecord:
        return cls(
            name=str(payload["name"]),
            value=payload["value"],
            step=payload["step"],
            unit=str(payload.get("unit", "unitless")),
            aggregation=str(payload.get("aggregation", "last")),
            scope=str(payload.get("scope", "learning")),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class LossResult:
    loss: float
    objective_scope: ObjectiveScope = ObjectiveScope()
    components: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metrics: tuple[MetricRecord, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    claims_not_made: tuple[str, ...] = LEARNING_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss", finite_number(self.loss, "loss"))
        if not isinstance(self.objective_scope, ObjectiveScope):
            raise TypeError("objective_scope must be ObjectiveScope")
        if not isinstance(self.components, Mapping):
            raise TypeError("loss components must be a mapping")
        components = {
            nonempty_string(name, "component name"): finite_number(value, "component")
            for name, value in self.components.items()
        }
        metrics = tuple(self.metrics)
        warnings = tuple(self.warnings)
        if any(not isinstance(item, MetricRecord) for item in metrics):
            raise TypeError("loss metrics must contain MetricRecord values")
        if any(not isinstance(item, LearningIssue) for item in warnings):
            raise TypeError("loss warnings must contain LearningIssue values")
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
            "loss": self.loss,
            "objective_scope": self.objective_scope.to_dict(),
            "components": dict(self.components),
            "metrics": [item.to_dict() for item in self.metrics],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LossResult:
        components = mapping(payload.get("components", {}), "components")
        return cls(
            loss=payload["loss"],
            objective_scope=ObjectiveScope.from_dict(
                mapping(payload.get("objective_scope", {}), "objective_scope")
            ),
            components=components,
            metrics=tuple(
                MetricRecord.from_dict(mapping(item, "metric"))
                for item in payload.get("metrics", ())
            ),
            warnings=tuple(
                LearningIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            claims_not_made=unique_strings(
                payload.get("claims_not_made", LEARNING_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


@dataclass(frozen=True)
class LearningStepResult:
    status: Literal["pass", "fail"]
    global_step_before: int
    global_step_after: int
    active_update_scope: UpdateScope = UpdateScope()
    active_objective_scope: ObjectiveScope = ObjectiveScope()
    loss: LossResult | None = None
    metrics: tuple[MetricRecord, ...] = ()
    changed_parameter_paths: tuple[str, ...] = ()
    unchanged_parameter_paths: tuple[str, ...] = ()
    blockers: tuple[LearningIssue, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    claims_not_made: tuple[str, ...] = LEARNING_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("learning step status must be pass or fail")
        nonnegative_int(self.global_step_before, "global_step_before")
        nonnegative_int(self.global_step_after, "global_step_after")
        if self.global_step_after < self.global_step_before:
            raise ValueError("global_step_after cannot precede global_step_before")
        if not isinstance(self.active_update_scope, UpdateScope):
            raise TypeError("active_update_scope must be UpdateScope")
        if not isinstance(self.active_objective_scope, ObjectiveScope):
            raise TypeError("active_objective_scope must be ObjectiveScope")
        if self.loss is not None and not isinstance(self.loss, LossResult):
            raise TypeError("loss must be LossResult when specified")
        metrics = tuple(self.metrics)
        blockers = tuple(self.blockers)
        warnings = tuple(self.warnings)
        changed = unique_strings(
            self.changed_parameter_paths, "changed_parameter_paths"
        )
        unchanged = unique_strings(
            self.unchanged_parameter_paths, "unchanged_parameter_paths"
        )
        if set(changed) & set(unchanged):
            raise ValueError("changed and unchanged parameter paths cannot overlap")
        if any(not isinstance(item, MetricRecord) for item in metrics):
            raise TypeError("metrics must contain MetricRecord values")
        if any(not isinstance(item, LearningIssue) for item in (*blockers, *warnings)):
            raise TypeError("blockers and warnings must contain LearningIssue values")
        if self.status == "pass" and blockers:
            raise ValueError("passing learning step cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing learning step requires blockers")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "changed_parameter_paths", changed)
        object.__setattr__(self, "unchanged_parameter_paths", unchanged)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "claims_not_made",
            unique_strings(self.claims_not_made, "claims_not_made"),
        )

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "global_step_before": self.global_step_before,
            "global_step_after": self.global_step_after,
            "active_update_scope": self.active_update_scope.to_dict(),
            "active_objective_scope": self.active_objective_scope.to_dict(),
            "loss": None if self.loss is None else self.loss.to_dict(),
            "metrics": [item.to_dict() for item in self.metrics],
            "changed_parameter_paths": list(self.changed_parameter_paths),
            "unchanged_parameter_paths": list(self.unchanged_parameter_paths),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningStepResult:
        loss = payload.get("loss")
        return cls(
            status=str(payload["status"]),
            global_step_before=payload["global_step_before"],
            global_step_after=payload["global_step_after"],
            active_update_scope=UpdateScope.from_dict(
                mapping(payload.get("active_update_scope", {}), "active_update_scope")
            ),
            active_objective_scope=ObjectiveScope.from_dict(
                mapping(
                    payload.get("active_objective_scope", {}),
                    "active_objective_scope",
                )
            ),
            loss=None if loss is None else LossResult.from_dict(mapping(loss, "loss")),
            metrics=tuple(
                MetricRecord.from_dict(mapping(item, "metric"))
                for item in payload.get("metrics", ())
            ),
            changed_parameter_paths=unique_strings(
                payload.get("changed_parameter_paths", ()), "changed_parameter_paths"
            ),
            unchanged_parameter_paths=unique_strings(
                payload.get("unchanged_parameter_paths", ()),
                "unchanged_parameter_paths",
            ),
            blockers=tuple(
                LearningIssue.from_dict(mapping(item, "blocker"))
                for item in payload.get("blockers", ())
            ),
            warnings=tuple(
                LearningIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            claims_not_made=unique_strings(
                payload.get("claims_not_made", LEARNING_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


@dataclass(frozen=True)
class LearningReport:
    status: Literal["pass", "fail"]
    config: LearningConfig
    state: LearningState
    latest_step: LearningStepResult | None = None
    metrics: tuple[MetricRecord, ...] = ()
    blockers: tuple[LearningIssue, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    claims_not_made: tuple[str, ...] = LEARNING_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("learning report status must be pass or fail")
        if not isinstance(self.config, LearningConfig) or not isinstance(
            self.state, LearningState
        ):
            raise TypeError("learning report requires LearningConfig and LearningState")
        if self.latest_step is not None and not isinstance(
            self.latest_step, LearningStepResult
        ):
            raise TypeError("latest_step must be LearningStepResult when specified")
        metrics = tuple(self.metrics)
        blockers = tuple(self.blockers)
        warnings = tuple(self.warnings)
        if any(not isinstance(item, MetricRecord) for item in metrics):
            raise TypeError("report metrics must contain MetricRecord values")
        if any(not isinstance(item, LearningIssue) for item in (*blockers, *warnings)):
            raise TypeError(
                "report blockers and warnings must contain LearningIssue values"
            )
        if self.status == "pass" and blockers:
            raise ValueError("passing learning report cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing learning report requires blockers")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "claims_not_made",
            unique_strings(self.claims_not_made, "claims_not_made"),
        )

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "latest_step": None
            if self.latest_step is None
            else self.latest_step.to_dict(),
            "metrics": [item.to_dict() for item in self.metrics],
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningReport:
        latest_step = payload.get("latest_step")
        return cls(
            status=str(payload["status"]),
            config=LearningConfig.from_dict(mapping(payload["config"], "config")),
            state=LearningState.from_dict(mapping(payload["state"], "state")),
            latest_step=(
                None
                if latest_step is None
                else LearningStepResult.from_dict(mapping(latest_step, "latest_step"))
            ),
            metrics=tuple(
                MetricRecord.from_dict(mapping(item, "metric"))
                for item in payload.get("metrics", ())
            ),
            blockers=tuple(
                LearningIssue.from_dict(mapping(item, "blocker"))
                for item in payload.get("blockers", ())
            ),
            warnings=tuple(
                LearningIssue.from_dict(mapping(item, "warning"))
                for item in payload.get("warnings", ())
            ),
            claims_not_made=unique_strings(
                payload.get("claims_not_made", LEARNING_CLAIMS_NOT_MADE),
                "claims_not_made",
            ),
        )


def canonical_learning_json(value: Mapping[str, Any]) -> bytes:
    """Encode a serializable learning artifact as deterministic finite JSON."""

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
