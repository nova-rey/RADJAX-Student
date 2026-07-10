from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from radjax_student.runtime.errors import RUNTIME_ERROR_CODES, RuntimeIssue
from radjax_student.runtime.models import (
    DeviceInventory,
    RuntimeCapabilityProfile,
    RuntimeConfig,
    RuntimeEnvironment,
    RuntimeStatus,
)


@dataclass(frozen=True)
class RuntimeReport:
    """Serializable runtime result without raw backend or device objects."""

    status: RuntimeStatus
    backend_id: str | None
    environment: RuntimeEnvironment | None
    device_inventory: DeviceInventory | None
    capabilities: RuntimeCapabilityProfile | None
    selected_policy: RuntimeConfig | None
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("runtime report status must be 'pass' or 'fail'")
        if self.backend_id is not None and not isinstance(self.backend_id, str):
            raise TypeError("backend_id must be a string when specified")
        if self.backend_id == "":
            raise ValueError("backend_id must be nonempty when specified")
        if self.environment is not None and not isinstance(
            self.environment,
            RuntimeEnvironment,
        ):
            raise TypeError("environment must be RuntimeEnvironment when specified")
        if self.device_inventory is not None and not isinstance(
            self.device_inventory,
            DeviceInventory,
        ):
            raise TypeError("device_inventory must be DeviceInventory when specified")
        if self.capabilities is not None and not isinstance(
            self.capabilities,
            RuntimeCapabilityProfile,
        ):
            raise TypeError(
                "capabilities must be RuntimeCapabilityProfile when specified"
            )
        if self.selected_policy is not None and not isinstance(
            self.selected_policy,
            RuntimeConfig,
        ):
            raise TypeError("selected_policy must be RuntimeConfig when specified")
        if (
            self.backend_id is not None
            and self.capabilities is not None
            and self.capabilities.backend_id != self.backend_id
        ):
            raise ValueError("report backend and capability backend must match")
        blockers = _issue_tuple(self.blockers, "blockers")
        warnings = _issue_tuple(self.warnings, "warnings")
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        unknown_blockers = [
            blocker.code
            for blocker in blockers
            if blocker.code not in RUNTIME_ERROR_CODES
        ]
        if unknown_blockers:
            raise ValueError(
                "runtime report contains unknown blocker codes: "
                + ", ".join(unknown_blockers)
            )
        if self.status == "pass" and blockers:
            raise ValueError("passing runtime report cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing runtime report must contain a blocker")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_not_made", claims)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend_id": self.backend_id,
            "environment": (
                None if self.environment is None else self.environment.to_dict()
            ),
            "device_inventory": (
                None
                if self.device_inventory is None
                else self.device_inventory.to_dict()
            ),
            "capabilities": (
                None if self.capabilities is None else self.capabilities.to_dict()
            ),
            "selected_policy": (
                None if self.selected_policy is None else self.selected_policy.to_dict()
            ),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeReport:
        environment = payload.get("environment")
        inventory = payload.get("device_inventory")
        capabilities = payload.get("capabilities")
        selected_policy = payload.get("selected_policy")
        return cls(
            status=str(payload["status"]),
            backend_id=(
                None
                if payload.get("backend_id") is None
                else str(payload["backend_id"])
            ),
            environment=(
                None
                if environment is None
                else RuntimeEnvironment.from_dict(_mapping(environment, "environment"))
            ),
            device_inventory=(
                None
                if inventory is None
                else DeviceInventory.from_dict(_mapping(inventory, "device_inventory"))
            ),
            capabilities=(
                None
                if capabilities is None
                else RuntimeCapabilityProfile.from_dict(
                    _mapping(capabilities, "capabilities")
                )
            ),
            selected_policy=(
                None
                if selected_policy is None
                else RuntimeConfig.from_dict(
                    _mapping(selected_policy, "selected_policy")
                )
            ),
            blockers=_issues(payload.get("blockers", ()), "blockers"),
            warnings=_issues(payload.get("warnings", ()), "warnings"),
            claims_not_made=_strings(
                payload.get("claims_not_made", ()),
                "claims_not_made",
            ),
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _issues(value: Any, name: str) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(RuntimeIssue.from_dict(_mapping(item, name)) for item in value)


def _issue_tuple(value: Any, name: str) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, RuntimeIssue) for item in result):
        raise TypeError(f"{name} must contain RuntimeIssue values")
    return result


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    result = _strings(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result
