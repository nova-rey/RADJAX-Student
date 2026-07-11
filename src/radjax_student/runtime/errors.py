from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
    "runtime_cpu_platform_required",
    "runtime_single_process_required",
    "runtime_single_device_required",
    "runtime_backend_initialization_failed",
    "runtime_device_selection_failed",
    "runtime_placement_failed",
    "runtime_execution_failed",
    "runtime_synchronization_failed",
    "runtime_smoke_result_mismatch",
    "runtime_teardown_failed",
    "runtime_smoke_internal_error",
    "execution_mode_unsupported",
    "execution_request_invalid",
    "execution_function_missing",
    "execution_context_mismatch",
    "execution_capability_missing",
    "execution_static_argument_invalid",
    "execution_donation_invalid",
    "execution_preparation_failed",
    "execution_compilation_failed",
    "execution_dispatch_failed",
    "execution_synchronization_failed",
    "execution_output_invalid",
    "execution_internal_error",
    "runtime_state_path_unsafe",
    "runtime_state_exists",
    "runtime_state_missing",
    "runtime_state_manifest_invalid",
    "runtime_state_schema_unsupported",
    "runtime_state_hash_mismatch",
    "runtime_state_size_mismatch",
    "runtime_state_config_invalid",
    "runtime_state_rng_invalid",
    "runtime_state_step_invalid",
    "runtime_state_resume_incompatible",
    "runtime_state_save_failed",
    "runtime_state_load_failed",
    "runtime_state_internal_error",
    "runtime_portability_platform_unavailable",
    "runtime_portability_backend_unavailable",
    "runtime_portability_device_unavailable",
    "runtime_portability_initialization_failed",
    "runtime_portability_placement_failed",
    "runtime_portability_execution_failed",
    "runtime_portability_synchronization_failed",
    "runtime_portability_result_mismatch",
    "runtime_portability_state_round_trip_failed",
    "runtime_portability_teardown_failed",
    "runtime_portability_internal_error",
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
    "runtime_cpu_platform_required",
    "runtime_single_process_required",
    "runtime_single_device_required",
    "runtime_backend_initialization_failed",
    "runtime_device_selection_failed",
    "runtime_placement_failed",
    "runtime_execution_failed",
    "runtime_synchronization_failed",
    "runtime_smoke_result_mismatch",
    "runtime_teardown_failed",
    "runtime_smoke_internal_error",
    "execution_mode_unsupported",
    "execution_request_invalid",
    "execution_function_missing",
    "execution_context_mismatch",
    "execution_capability_missing",
    "execution_static_argument_invalid",
    "execution_donation_invalid",
    "execution_preparation_failed",
    "execution_compilation_failed",
    "execution_dispatch_failed",
    "execution_synchronization_failed",
    "execution_output_invalid",
    "execution_internal_error",
    "runtime_state_path_unsafe",
    "runtime_state_exists",
    "runtime_state_missing",
    "runtime_state_manifest_invalid",
    "runtime_state_schema_unsupported",
    "runtime_state_hash_mismatch",
    "runtime_state_size_mismatch",
    "runtime_state_config_invalid",
    "runtime_state_rng_invalid",
    "runtime_state_step_invalid",
    "runtime_state_resume_incompatible",
    "runtime_state_save_failed",
    "runtime_state_load_failed",
    "runtime_state_internal_error",
    "runtime_portability_platform_unavailable",
    "runtime_portability_backend_unavailable",
    "runtime_portability_device_unavailable",
    "runtime_portability_initialization_failed",
    "runtime_portability_placement_failed",
    "runtime_portability_execution_failed",
    "runtime_portability_synchronization_failed",
    "runtime_portability_result_mismatch",
    "runtime_portability_state_round_trip_failed",
    "runtime_portability_teardown_failed",
    "runtime_portability_internal_error",
    "runtime_internal_error",
)


@dataclass(frozen=True)
class RuntimeIssue:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

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
