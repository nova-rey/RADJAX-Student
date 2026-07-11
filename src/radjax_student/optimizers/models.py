"""Immutable architecture-independent optimizer contract models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from radjax_student.architecture import ParameterCatalog
from radjax_student.learning import MetricRecord, ResolvedUpdateSelection
from radjax_student.optimizers._json import (
    finite_number,
    freeze_mapping,
    json_value,
    mapping,
    positive_int,
    strings,
)
from radjax_student.optimizers.errors import OptimizerIssue

OPTIMIZER_CONFIG_SCHEMA_VERSION = "optimizer_config.v1"
OPTIMIZER_STATE_SCHEMA_VERSION = "optimizer_state.v1"
GRADIENT_CLIP_MODES: tuple[str, ...] = (
    "disabled",
    "global_norm",
    "per_parameter_norm",
    "value",
    "plugin_defined",
)
WEIGHT_DECAY_MODES: tuple[str, ...] = (
    "disabled",
    "coupled",
    "decoupled",
    "plugin_defined",
)
OPTIMIZER_STATE_ROLES: tuple[str, ...] = (
    "accumulator",
    "first_moment",
    "momentum",
    "other",
    "second_moment",
    "step_counter",
)
OPTIMIZER_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "gradient_not_computed",
    "optax_not_invoked",
    "schedule_not_executed",
    "mixed_precision_not_proven",
    "distributed_optimization_not_proven",
    "learning_step_not_run",
    "training_loop_not_run",
)


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_id: str
    schema_version: str = OPTIMIZER_CONFIG_SCHEMA_VERSION
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    weight_decay_mode: str = "disabled"
    gradient_clip_mode: str = "disabled"
    gradient_clip: float | None = None
    epsilon: float | None = None
    momentum: float | None = None
    schedule_reference: str | None = None
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer_id, str) or not self.optimizer_id:
            raise ValueError("optimizer_id must be a nonempty string")
        if self.schema_version != OPTIMIZER_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported optimizer config schema version")
        if finite_number(self.learning_rate, "learning_rate", nonnegative=True) <= 0:
            raise ValueError("learning_rate must be positive")
        finite_number(self.weight_decay, "weight_decay", nonnegative=True)
        if self.weight_decay_mode not in WEIGHT_DECAY_MODES:
            raise ValueError("weight_decay_mode is unsupported")
        if self.gradient_clip_mode not in GRADIENT_CLIP_MODES:
            raise ValueError("gradient_clip_mode is unsupported")
        if (
            self.gradient_clip is not None
            and finite_number(self.gradient_clip, "gradient_clip", nonnegative=True)
            <= 0
        ):
            raise ValueError("gradient_clip must be positive when specified")
        if self.gradient_clip_mode == "disabled" and self.gradient_clip is not None:
            raise ValueError("disabled clipping cannot carry gradient_clip")
        if self.gradient_clip_mode != "disabled" and self.gradient_clip is None:
            raise ValueError("enabled clipping requires gradient_clip")
        if (
            self.epsilon is not None
            and finite_number(self.epsilon, "epsilon", nonnegative=True) <= 0
        ):
            raise ValueError("epsilon must be positive when specified")
        if self.momentum is not None:
            momentum = finite_number(self.momentum, "momentum", nonnegative=True)
            if momentum >= 1:
                raise ValueError("momentum must be less than one")
        if self.schedule_reference is not None and (
            not isinstance(self.schedule_reference, str) or not self.schedule_reference
        ):
            raise ValueError(
                "schedule_reference must be a nonempty string when specified"
            )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_id": self.optimizer_id,
            "schema_version": self.schema_version,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "weight_decay_mode": self.weight_decay_mode,
            "gradient_clip_mode": self.gradient_clip_mode,
            "gradient_clip": self.gradient_clip,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "schedule_reference": self.schedule_reference,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OptimizerConfig:
        return cls(
            optimizer_id=str(payload["optimizer_id"]),
            schema_version=str(
                payload.get("schema_version", OPTIMIZER_CONFIG_SCHEMA_VERSION)
            ),
            learning_rate=payload.get("learning_rate", 0.001),
            weight_decay=payload.get("weight_decay", 0.0),
            weight_decay_mode=str(payload.get("weight_decay_mode", "disabled")),
            gradient_clip_mode=str(payload.get("gradient_clip_mode", "disabled")),
            gradient_clip=payload.get("gradient_clip"),
            epsilon=payload.get("epsilon"),
            momentum=payload.get("momentum"),
            schedule_reference=payload.get("schedule_reference"),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class OptimizerCapabilityProfile:
    optimizer_id: str
    version: int
    capabilities: tuple[str, ...]
    non_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer_id, str) or not self.optimizer_id:
            raise ValueError("optimizer_id must be a nonempty string")
        positive_int(self.version, "version")
        capabilities, non_capabilities = (
            strings(self.capabilities, "capabilities", sort=True),
            strings(self.non_capabilities, "non_capabilities", sort=True),
        )
        if set(capabilities) & set(non_capabilities):
            raise ValueError("capabilities and non_capabilities cannot overlap")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "non_capabilities", non_capabilities)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_id": self.optimizer_id,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "non_capabilities": list(self.non_capabilities),
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class GradientTree:
    parameter_paths: tuple[str, ...]
    values: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        paths = strings(self.parameter_paths, "parameter_paths", sort=True)
        if not paths:
            raise ValueError("gradient tree requires parameter paths")
        object.__setattr__(self, "parameter_paths", paths)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_paths": list(self.parameter_paths),
            "values_present": self.values is not None,
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class OptimizerState:
    optimizer_id: str
    parameter_paths: tuple[str, ...]
    step: int = 0
    schema_version: str = OPTIMIZER_STATE_SCHEMA_VERSION
    state_structure: Mapping[str, Any] = MappingProxyType({})
    backend_state: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = MappingProxyType({})
    claims_not_made: tuple[str, ...] = OPTIMIZER_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer_id, str) or not self.optimizer_id:
            raise ValueError("optimizer_id must be a nonempty string")
        if self.schema_version != OPTIMIZER_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported optimizer state schema version")
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("step must be a nonnegative integer")
        object.__setattr__(
            self,
            "parameter_paths",
            strings(self.parameter_paths, "parameter_paths", sort=True),
        )
        object.__setattr__(
            self, "state_structure", freeze_mapping(self.state_structure)
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(
            self, "claims_not_made", strings(self.claims_not_made, "claims_not_made")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "optimizer_id": self.optimizer_id,
            "step": self.step,
            "parameter_paths": list(self.parameter_paths),
            "state_structure": json_value(self.state_structure),
            "backend_state_present": self.backend_state is not None,
            "metadata": json_value(self.metadata),
            "claims_not_made": list(self.claims_not_made),
        }


@dataclass(frozen=True)
class OptimizerStateDescriptor:
    optimizer_id: str
    step: int
    tracked_parameter_paths: tuple[str, ...]
    state_roles: tuple[str, ...]
    state_count: int
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer_id, str) or not self.optimizer_id:
            raise ValueError("optimizer_id must be a nonempty string")
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("step must be a nonnegative integer")
        if (
            not isinstance(self.state_count, int)
            or isinstance(self.state_count, bool)
            or self.state_count < 0
        ):
            raise ValueError("state_count must be a nonnegative integer")
        roles = strings(self.state_roles, "state_roles", sort=True)
        if any(role not in OPTIMIZER_STATE_ROLES for role in roles):
            raise ValueError("state role is unsupported")
        object.__setattr__(
            self,
            "tracked_parameter_paths",
            strings(self.tracked_parameter_paths, "tracked_parameter_paths", sort=True),
        )
        object.__setattr__(self, "state_roles", roles)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_id": self.optimizer_id,
            "step": self.step,
            "tracked_parameter_paths": list(self.tracked_parameter_paths),
            "state_roles": list(self.state_roles),
            "state_count": self.state_count,
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class OptimizerInitRequest:
    config: OptimizerConfig
    parameter_catalog: ParameterCatalog
    resolved_update_selection: ResolvedUpdateSelection
    runtime_metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config, OptimizerConfig)
            or not isinstance(self.parameter_catalog, ParameterCatalog)
            or not isinstance(self.resolved_update_selection, ResolvedUpdateSelection)
        ):
            raise TypeError(
                "optimizer initialization requires config, catalog, and resolved "
                "selection"
            )
        _selection_in_catalog(self.resolved_update_selection, self.parameter_catalog)
        object.__setattr__(
            self, "runtime_metadata", freeze_mapping(self.runtime_metadata)
        )


@dataclass(frozen=True)
class OptimizerInitResult:
    optimizer_state: OptimizerState
    state_metadata: Mapping[str, Any] = MappingProxyType({})
    warnings: tuple[OptimizerIssue, ...] = ()
    claims_not_made: tuple[str, ...] = OPTIMIZER_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer_state, OptimizerState):
            raise TypeError("optimizer_state must be OptimizerState")
        _findings(self.warnings, "warnings")
        object.__setattr__(self, "state_metadata", freeze_mapping(self.state_metadata))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self, "claims_not_made", strings(self.claims_not_made, "claims_not_made")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_state": self.optimizer_state.to_dict(),
            "state_metadata": json_value(self.state_metadata),
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }


@dataclass(frozen=True)
class OptimizerUpdateRequest:
    gradients: GradientTree
    optimizer_state: OptimizerState
    config: OptimizerConfig
    resolved_update_selection: ResolvedUpdateSelection
    learning_step: int
    parameters: Any = field(default=None, repr=False, compare=False)
    schedule_values: Mapping[str, Any] = MappingProxyType({})
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if (
            not isinstance(self.gradients, GradientTree)
            or not isinstance(self.optimizer_state, OptimizerState)
            or not isinstance(self.config, OptimizerConfig)
            or not isinstance(self.resolved_update_selection, ResolvedUpdateSelection)
        ):
            raise TypeError("optimizer update request members are invalid")
        if self.optimizer_state.optimizer_id != self.config.optimizer_id:
            raise ValueError("optimizer state and configuration IDs must match")
        if (
            not isinstance(self.learning_step, int)
            or isinstance(self.learning_step, bool)
            or self.learning_step < 0
        ):
            raise ValueError("learning_step must be a nonnegative integer")
        if tuple(self.optimizer_state.parameter_paths) != tuple(
            self.gradients.parameter_paths
        ):
            raise ValueError("optimizer state and gradient paths must match")
        selected = set(self.resolved_update_selection.selected_parameter_paths)
        if not selected <= set(self.gradients.parameter_paths):
            raise ValueError("selected update paths must exist in gradients")
        object.__setattr__(
            self, "schedule_values", freeze_mapping(self.schedule_values)
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gradients": self.gradients.to_dict(),
            "optimizer_state": self.optimizer_state.to_dict(),
            "config": self.config.to_dict(),
            "resolved_update_selection": self.resolved_update_selection.to_dict(),
            "learning_step": self.learning_step,
            "parameters_present": self.parameters is not None,
            "schedule_values": json_value(self.schedule_values),
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class ParameterUpdate:
    parameter_path: str
    applied: bool
    update_norm: float | None = None
    clipped: bool = False
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_path, str) or not self.parameter_path:
            raise ValueError("parameter_path must be nonempty")
        if not isinstance(self.applied, bool) or not isinstance(self.clipped, bool):
            raise TypeError("applied and clipped must be booleans")
        if self.update_norm is not None:
            finite_number(self.update_norm, "update_norm", nonnegative=True)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_path": self.parameter_path,
            "applied": self.applied,
            "update_norm": self.update_norm,
            "clipped": self.clipped,
            "metadata": json_value(self.metadata),
        }


@dataclass(frozen=True)
class OptimizerUpdateResult:
    updated_optimizer_state: OptimizerState
    parameter_updates: tuple[ParameterUpdate, ...]
    changed_parameter_paths: tuple[str, ...]
    unchanged_parameter_paths: tuple[str, ...]
    updated_parameters: Any = field(default=None, repr=False, compare=False)
    update_metadata: Mapping[str, Any] = MappingProxyType({})
    metrics: tuple[MetricRecord, ...] = ()
    warnings: tuple[OptimizerIssue, ...] = ()
    claims_not_made: tuple[str, ...] = OPTIMIZER_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if not isinstance(self.updated_optimizer_state, OptimizerState):
            raise TypeError("updated_optimizer_state must be OptimizerState")
        updates = tuple(self.parameter_updates)
        if any(not isinstance(item, ParameterUpdate) for item in updates):
            raise TypeError("parameter_updates must contain ParameterUpdate values")
        changed, unchanged = (
            strings(self.changed_parameter_paths, "changed_parameter_paths", sort=True),
            strings(
                self.unchanged_parameter_paths, "unchanged_parameter_paths", sort=True
            ),
        )
        if set(changed) & set(unchanged):
            raise ValueError("changed and unchanged parameter paths cannot overlap")
        if any(not isinstance(item, MetricRecord) for item in self.metrics):
            raise TypeError("metrics must contain MetricRecord values")
        _findings(self.warnings, "warnings")
        object.__setattr__(self, "parameter_updates", updates)
        object.__setattr__(self, "changed_parameter_paths", changed)
        object.__setattr__(self, "unchanged_parameter_paths", unchanged)
        object.__setattr__(
            self, "update_metadata", freeze_mapping(self.update_metadata)
        )
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self, "claims_not_made", strings(self.claims_not_made, "claims_not_made")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_parameters_present": self.updated_parameters is not None,
            "updated_optimizer_state": self.updated_optimizer_state.to_dict(),
            "parameter_updates": [item.to_dict() for item in self.parameter_updates],
            "changed_parameter_paths": list(self.changed_parameter_paths),
            "unchanged_parameter_paths": list(self.unchanged_parameter_paths),
            "update_metadata": json_value(self.update_metadata),
            "metrics": [item.to_dict() for item in self.metrics],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }


def canonical_optimizer_json(value: Mapping[str, Any]) -> bytes:
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


def _selection_in_catalog(
    selection: ResolvedUpdateSelection, catalog: ParameterCatalog
) -> None:
    unknown = set(selection.selected_parameter_paths) - set(catalog.paths)
    if unknown:
        raise ValueError("resolved selection references unknown catalog paths")


def _findings(items: tuple[OptimizerIssue, ...], name: str) -> None:
    if any(not isinstance(item, OptimizerIssue) for item in items):
        raise TypeError(f"{name} must contain OptimizerIssue values")
