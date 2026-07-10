from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from radjax_student.runtime.models import freeze_json_mapping, json_value

RuntimeErrorCode: TypeAlias = Literal[
    "runtime_backend_not_found",
    "runtime_backend_unavailable",
    "runtime_backend_duplicate",
    "runtime_backend_ineligible",
    "requested_platform_unavailable",
    "runtime_capability_missing",
    "runtime_policy_unsupported",
    "runtime_initialization_failed",
    "runtime_configuration_invalid",
    "runtime_environment_incompatible",
    "runtime_fallback_disallowed",
    "runtime_selection_ambiguous",
    "runtime_selection_internal_error",
    "runtime_internal_error",
]

RUNTIME_ERROR_CODES: tuple[str, ...] = (
    "runtime_backend_not_found",
    "runtime_backend_unavailable",
    "runtime_backend_duplicate",
    "runtime_backend_ineligible",
    "requested_platform_unavailable",
    "runtime_capability_missing",
    "runtime_policy_unsupported",
    "runtime_initialization_failed",
    "runtime_configuration_invalid",
    "runtime_environment_incompatible",
    "runtime_fallback_disallowed",
    "runtime_selection_ambiguous",
    "runtime_selection_internal_error",
    "runtime_internal_error",
)


@dataclass(frozen=True)
class RuntimeIssue:
    code: str
    message: str
    details: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("runtime issue code and message must be nonempty")
        object.__setattr__(self, "details", freeze_json_mapping(self.details))

    @classmethod
    def create(cls, code: str, message: str, **details: Any) -> RuntimeIssue:
        return cls(code=code, message=message, details=details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": json_value(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeIssue:
        details = payload.get("details", {})
        if not isinstance(details, Mapping):
            raise TypeError("runtime issue details must be a mapping")
        return cls(
            code=str(payload["code"]),
            message=str(payload["message"]),
            details=details,
        )


class RuntimeContractError(Exception):
    """Structured runtime boundary failure with a stable public code."""

    def __init__(
        self,
        code: RuntimeErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in RUNTIME_ERROR_CODES:
            raise ValueError(f"unknown runtime error code: {code}")
        self.issue = RuntimeIssue(
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
