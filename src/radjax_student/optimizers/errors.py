"""Structured optimizer-boundary failures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from radjax_student.optimizers._json import freeze_mapping, json_value, mapping

OptimizerErrorCode: TypeAlias = Literal[
    "optimizer_backend_not_found",
    "optimizer_backend_duplicate",
    "optimizer_config_invalid",
    "optimizer_capability_missing",
    "optimizer_state_invalid",
    "optimizer_state_parameter_mismatch",
    "optimizer_gradient_structure_invalid",
    "optimizer_gradient_nonfinite",
    "optimizer_update_scope_invalid",
    "optimizer_update_failed",
    "optimizer_state_transition_failed",
    "optimizer_internal_error",
]
OPTIMIZER_ERROR_CODES: tuple[str, ...] = (
    "optimizer_backend_not_found",
    "optimizer_backend_duplicate",
    "optimizer_config_invalid",
    "optimizer_capability_missing",
    "optimizer_state_invalid",
    "optimizer_state_parameter_mismatch",
    "optimizer_gradient_structure_invalid",
    "optimizer_gradient_nonfinite",
    "optimizer_update_scope_invalid",
    "optimizer_update_failed",
    "optimizer_state_transition_failed",
    "optimizer_internal_error",
)


@dataclass(frozen=True)
class OptimizerIssue:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not self.code
            or not isinstance(self.message, str)
            or not self.message
        ):
            raise ValueError("optimizer issue code and message must be nonempty")
        object.__setattr__(self, "details", freeze_mapping(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": json_value(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OptimizerIssue:
        return cls(
            str(payload["code"]),
            str(payload["message"]),
            mapping(payload.get("details", {}), "details"),
        )


class OptimizerContractError(Exception):
    def __init__(
        self,
        code: OptimizerErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in OPTIMIZER_ERROR_CODES:
            raise ValueError(f"unknown optimizer error code: {code}")
        self.issue = OptimizerIssue(code, message, {} if details is None else details)
        super().__init__(f"{code}: {message}")

    @property
    def code(self) -> str:
        return self.issue.code

    @property
    def details(self) -> Mapping[str, Any]:
        return self.issue.details
