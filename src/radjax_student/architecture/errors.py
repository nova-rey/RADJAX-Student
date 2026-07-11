"""Structured failures for architecture-plugin contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from radjax_student.architecture._json import freeze_mapping, json_value, mapping

ArchitectureErrorCode: TypeAlias = Literal[
    "architecture_plugin_not_found",
    "architecture_plugin_duplicate",
    "architecture_config_invalid",
    "architecture_capability_missing",
    "architecture_parameter_catalog_invalid",
    "architecture_parameter_path_unknown",
    "architecture_update_scope_unsupported",
    "architecture_update_scope_resolution_failed",
    "architecture_objective_scope_unsupported",
    "architecture_objective_scope_resolution_failed",
    "architecture_batch_incompatible",
    "architecture_initialization_failed",
    "architecture_forward_failed",
    "architecture_internal_error",
]

ARCHITECTURE_ERROR_CODES: tuple[str, ...] = (
    "architecture_plugin_not_found",
    "architecture_plugin_duplicate",
    "architecture_config_invalid",
    "architecture_capability_missing",
    "architecture_parameter_catalog_invalid",
    "architecture_parameter_path_unknown",
    "architecture_update_scope_unsupported",
    "architecture_update_scope_resolution_failed",
    "architecture_objective_scope_unsupported",
    "architecture_objective_scope_resolution_failed",
    "architecture_batch_incompatible",
    "architecture_initialization_failed",
    "architecture_forward_failed",
    "architecture_internal_error",
)


@dataclass(frozen=True)
class ArchitectureIssue:
    code: str
    message: str
    details: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("architecture issue code must be a nonempty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("architecture issue message must be a nonempty string")
        object.__setattr__(self, "details", freeze_mapping(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": json_value(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureIssue:
        return cls(
            code=str(payload["code"]),
            message=str(payload["message"]),
            details=mapping(payload.get("details", {}), "details"),
        )


class ArchitectureContractError(Exception):
    """A stable architecture-boundary failure with structured details."""

    def __init__(
        self,
        code: ArchitectureErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in ARCHITECTURE_ERROR_CODES:
            raise ValueError(f"unknown architecture error code: {code}")
        self.issue = ArchitectureIssue(
            code, message, {} if details is None else details
        )
        super().__init__(f"{code}: {message}")

    @property
    def code(self) -> str:
        return self.issue.code

    @property
    def details(self) -> Mapping[str, Any]:
        return self.issue.details
