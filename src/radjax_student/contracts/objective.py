"""Architecture-resolved objective surface selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from radjax_student.contracts._json import freeze_json_mapping, mapping, unique_strings
from radjax_student.contracts.scopes import ObjectiveScope


@dataclass(frozen=True)
class ResolvedObjectiveSelection:
    scope: ObjectiveScope
    surface_id: str
    required_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scope, ObjectiveScope)
            or not isinstance(self.surface_id, str)
            or not self.surface_id
        ):
            raise ValueError("resolved objective selection is invalid")
        object.__setattr__(
            self,
            "required_capabilities",
            tuple(
                sorted(
                    unique_strings(self.required_capabilities, "required_capabilities")
                )
            ),
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "surface_id": self.surface_id,
            "required_capabilities": list(self.required_capabilities),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResolvedObjectiveSelection:
        return cls(
            ObjectiveScope.from_dict(mapping(payload["scope"], "scope")),
            str(payload["surface_id"]),
            tuple(payload.get("required_capabilities", ())),
            mapping(payload.get("metadata", {}), "metadata"),
        )


__all__ = ["ResolvedObjectiveSelection"]
