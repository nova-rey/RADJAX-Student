"""Architecture-neutral declarations of update and objective scope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.learning._json import (
    freeze_json_mapping,
    json_value,
    mapping,
    optional_string,
    unique_strings,
)
from radjax_student.learning.errors import LearningContractError

UpdateScopeKind = Literal[
    "whole_student",
    "named_region",
    "parameter_paths",
    "plugin_defined",
]
ObjectiveScopeKind = Literal[
    "final_output",
    "whole_student",
    "named_region",
    "intermediate_surface",
    "plugin_defined",
]

UPDATE_SCOPE_KINDS: tuple[UpdateScopeKind, ...] = (
    "whole_student",
    "named_region",
    "parameter_paths",
    "plugin_defined",
)
OBJECTIVE_SCOPE_KINDS: tuple[ObjectiveScopeKind, ...] = (
    "final_output",
    "whole_student",
    "named_region",
    "intermediate_surface",
    "plugin_defined",
)


@dataclass(frozen=True)
class UpdateScope:
    """Parameter-change intent; architecture plugins interpret region identities."""

    kind: UpdateScopeKind = "whole_student"
    region_id: str | None = None
    parameter_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.kind not in UPDATE_SCOPE_KINDS:
            raise LearningContractError(
                "learning_update_scope_invalid",
                "update scope kind is unsupported",
                details={"kind": self.kind},
            )
        region_id = optional_string(self.region_id, "region_id")
        paths = unique_strings(self.parameter_paths, "parameter_paths")
        if any(not _parameter_path_is_stable(path) for path in paths):
            raise LearningContractError(
                "learning_update_scope_invalid",
                "parameter_paths must contain stable relative paths",
            )
        if self.kind == "whole_student" and (region_id is not None or paths):
            raise LearningContractError(
                "learning_update_scope_invalid",
                "whole_student update scope cannot declare a region or parameter paths",
            )
        if self.kind in ("named_region", "plugin_defined") and region_id is None:
            raise LearningContractError(
                "learning_update_scope_invalid",
                "named or plugin-defined update scope requires region_id",
            )
        if self.kind == "parameter_paths" and (region_id is not None or not paths):
            raise LearningContractError(
                "learning_update_scope_invalid",
                "parameter_paths update scope requires paths and no region_id",
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("update scope metadata must be a mapping")
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "parameter_paths", paths)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "region_id": self.region_id,
            "parameter_paths": list(self.parameter_paths),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UpdateScope:
        return cls(
            kind=str(payload.get("kind", "whole_student")),
            region_id=payload.get("region_id"),
            parameter_paths=unique_strings(
                payload.get("parameter_paths", ()), "parameter_paths"
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ObjectiveScope:
    """Learning-signal observation intent, independent of parameter updates."""

    kind: ObjectiveScopeKind = "final_output"
    target_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.kind not in OBJECTIVE_SCOPE_KINDS:
            raise LearningContractError(
                "learning_objective_scope_invalid",
                "objective scope kind is unsupported",
                details={"kind": self.kind},
            )
        target_id = optional_string(self.target_id, "target_id")
        if self.kind in ("final_output", "whole_student") and target_id is not None:
            raise LearningContractError(
                "learning_objective_scope_invalid",
                "default objective scopes cannot declare target_id",
            )
        if (
            self.kind in ("named_region", "intermediate_surface", "plugin_defined")
            and target_id is None
        ):
            raise LearningContractError(
                "learning_objective_scope_invalid",
                "scoped objective requires target_id",
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("objective scope metadata must be a mapping")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_id": self.target_id,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveScope:
        return cls(
            kind=str(payload.get("kind", "final_output")),
            target_id=payload.get("target_id"),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class ResolvedUpdateSelection:
    """Architecture-resolved stable paths, not a parameter tree or update mask."""

    selection_id: str
    selected_parameter_paths: tuple[str, ...]
    excluded_parameter_paths: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.selection_id, str) or not self.selection_id:
            raise LearningContractError(
                "learning_scope_resolution_failed",
                "selection_id must be a nonempty string",
            )
        selected = unique_strings(
            self.selected_parameter_paths, "selected_parameter_paths"
        )
        excluded = unique_strings(
            self.excluded_parameter_paths, "excluded_parameter_paths"
        )
        if not selected:
            raise LearningContractError(
                "learning_scope_resolution_failed",
                "resolved update selection must contain selected parameter paths",
            )
        if any(not _parameter_path_is_stable(path) for path in (*selected, *excluded)):
            raise LearningContractError(
                "learning_scope_resolution_failed",
                "resolved parameter paths must be stable relative paths",
            )
        overlap = sorted(set(selected) & set(excluded))
        if overlap:
            raise LearningContractError(
                "learning_scope_resolution_failed",
                "selected and excluded parameter paths cannot overlap",
                details={"overlap": overlap},
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("resolved update selection metadata must be a mapping")
        object.__setattr__(self, "selected_parameter_paths", selected)
        object.__setattr__(self, "excluded_parameter_paths", excluded)
        object.__setattr__(
            self, "capabilities", unique_strings(self.capabilities, "capabilities")
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "selected_parameter_paths": list(self.selected_parameter_paths),
            "excluded_parameter_paths": list(self.excluded_parameter_paths),
            "capabilities": list(self.capabilities),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResolvedUpdateSelection:
        return cls(
            selection_id=str(payload["selection_id"]),
            selected_parameter_paths=unique_strings(
                payload.get("selected_parameter_paths", ()),
                "selected_parameter_paths",
            ),
            excluded_parameter_paths=unique_strings(
                payload.get("excluded_parameter_paths", ()),
                "excluded_parameter_paths",
            ),
            capabilities=unique_strings(
                payload.get("capabilities", ()), "capabilities"
            ),
            metadata=mapping(payload.get("metadata", {}), "metadata"),
        )


def _parameter_path_is_stable(path: str) -> bool:
    return bool(path) and not path.startswith(("/", ".")) and "//" not in path
