"""Structured, backend-neutral failures for the generic learning contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from radjax_student.learning._json import freeze_json_mapping, json_value, mapping

LearningErrorCode: TypeAlias = Literal[
    "learning_config_invalid",
    "learning_state_invalid",
    "learning_batch_invalid",
    "learning_update_scope_invalid",
    "learning_objective_scope_invalid",
    "learning_scope_resolution_failed",
    "learning_scope_capability_missing",
    "learning_step_failed",
    "learning_checkpoint_policy_invalid",
    "learning_internal_error",
]

LEARNING_ERROR_CODES: tuple[str, ...] = (
    "learning_config_invalid",
    "learning_state_invalid",
    "learning_batch_invalid",
    "learning_update_scope_invalid",
    "learning_objective_scope_invalid",
    "learning_scope_resolution_failed",
    "learning_scope_capability_missing",
    "learning_step_failed",
    "learning_checkpoint_policy_invalid",
    "learning_internal_error",
)


@dataclass(frozen=True)
class LearningIssue:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("learning issue code must be a nonempty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("learning issue message must be a nonempty string")
        object.__setattr__(self, "details", freeze_json_mapping(self.details))

    @classmethod
    def create(cls, code: str, message: str, **details: Any) -> LearningIssue:
        return cls(code=code, message=message, details=details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": json_value(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningIssue:
        return cls(
            code=str(payload["code"]),
            message=str(payload["message"]),
            details=mapping(payload.get("details", {}), "details"),
        )


class LearningContractError(Exception):
    """A stable public learning-contract failure with structured details."""

    def __init__(
        self,
        code: LearningErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in LEARNING_ERROR_CODES:
            raise ValueError(f"unknown learning error code: {code}")
        self.issue = LearningIssue(
            code=code,
            message=message,
            details={} if details is None else details,
        )
        super().__init__(f"{code}: {message}")

    @property
    def code(self) -> str:
        return self.issue.code

    @property
    def details(self) -> Mapping[str, Any]:
        return self.issue.details

    def to_dict(self) -> dict[str, Any]:
        return self.issue.to_dict()
