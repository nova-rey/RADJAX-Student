"""Dependency-neutral identity and surface contracts for objective plugins."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from radjax_student.contracts._json import (
    freeze_json_mapping,
    json_value,
    mapping,
    nonempty_string,
    unique_strings,
)
from radjax_student.contracts.scopes import ObjectiveScope

OBJECTIVE_CAPABILITY_SCHEMA_VERSION = "radjax.objective.capability.v1"
OBJECTIVE_CONFIG_SCHEMA_VERSION = "radjax.objective.config.v1"
OBJECTIVE_EXECUTION_DESCRIPTOR_SCHEMA_VERSION = "radjax.objective.execution.v1"
_STABLE_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_STABLE_TOKEN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ObjectiveContractError(ValueError):
    """Stable public failure for canonical objective contract validation."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ):
        self.code = code
        self.details = MappingProxyType(dict(details or {}))
        super().__init__(f"{code}: {message}")


def canonical_objective_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def objective_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_objective_json(value)).hexdigest()


def _stable_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _STABLE_ID.fullmatch(value):
        raise ObjectiveContractError(
            "objective_identity_mismatch",
            f"{name} must be a canonical dotted stable ID",
        )
    return value


def _stable_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _STABLE_TOKEN.fullmatch(value):
        raise ObjectiveContractError(
            "objective_identity_mismatch", f"{name} must be a canonical stable token"
        )
    return value


def _version(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.isdecimal() or value == "0":
        raise ObjectiveContractError(
            "objective_identity_mismatch", f"{name} must be a positive decimal string"
        )
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ObjectiveContractError(
            "objective_config_invalid", f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_fields(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ObjectiveContractError(
            "objective_config_invalid",
            f"{name} fields are missing or unknown",
        )


@dataclass(frozen=True)
class ObjectiveIdentity:
    objective_id: str
    objective_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "objective_id", _stable_id(self.objective_id, "objective_id")
        )
        object.__setattr__(
            self,
            "objective_version",
            _version(self.objective_version, "objective_version"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "objective_id": self.objective_id,
            "objective_version": self.objective_version,
        }

    @property
    def digest(self) -> str:
        return objective_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveIdentity:
        _strict_fields(payload, {"objective_id", "objective_version"}, "identity")
        return cls(str(payload["objective_id"]), str(payload["objective_version"]))


@dataclass(frozen=True)
class ResolvedObjectiveSelection:
    """One architecture-owned surface selection, never a caller interpretation."""

    scope: ObjectiveScope
    surface_id: str
    surface_role: str = "prediction"
    required_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ObjectiveScope):
            raise ObjectiveContractError(
                "objective_surface_missing",
                "resolved objective selection requires ObjectiveScope",
            )
        nonempty_string(self.surface_id, "surface_id")
        _stable_token(self.surface_role, "surface_role")
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
            "surface_role": self.surface_role,
            "required_capabilities": list(self.required_capabilities),
            "metadata": json_value(self.metadata),
        }

    @property
    def digest(self) -> str:
        return objective_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResolvedObjectiveSelection:
        _strict_fields(
            payload,
            {
                "scope",
                "surface_id",
                "surface_role",
                "required_capabilities",
                "metadata",
            },
            "resolved objective selection",
        )
        return cls(
            ObjectiveScope.from_dict(mapping(payload["scope"], "scope")),
            str(payload["surface_id"]),
            str(payload["surface_role"]),
            tuple(payload["required_capabilities"]),
            mapping(payload["metadata"], "metadata"),
        )


@dataclass(frozen=True)
class ObjectiveCapabilityProfile:
    identity: ObjectiveIdentity
    supported_execution_capabilities: tuple[str, ...]
    required_surface_roles: tuple[str, ...]
    target_requirements: tuple[str, ...]
    metric_schema_id: str
    metric_names: tuple[str, ...]
    capability_schema_version: str = OBJECTIVE_CAPABILITY_SCHEMA_VERSION
    non_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ObjectiveIdentity):
            raise ObjectiveContractError(
                "objective_plugin_invalid",
                "capability profile requires ObjectiveIdentity",
            )
        if self.capability_schema_version != OBJECTIVE_CAPABILITY_SCHEMA_VERSION:
            raise ObjectiveContractError(
                "objective_capability_mismatch",
                "unsupported objective capability schema",
            )
        try:
            capabilities = tuple(
                sorted(
                    unique_strings(
                        self.supported_execution_capabilities, "capabilities"
                    )
                )
            )
        except (TypeError, ValueError) as error:
            raise ObjectiveContractError(
                "objective_capability_mismatch",
                "objective capabilities must be unique stable strings",
            ) from error
        if not capabilities:
            raise ObjectiveContractError(
                "objective_capability_missing",
                "objective profile requires capabilities",
            )
        try:
            roles = tuple(
                sorted(
                    _stable_token(item, "required_surface_roles")
                    for item in self.required_surface_roles
                )
            )
        except (TypeError, ValueError, ObjectiveContractError) as error:
            raise ObjectiveContractError(
                "objective_surface_missing",
                "objective surface roles must be unique stable strings",
            ) from error
        if not roles or len(roles) != len(set(roles)):
            raise ObjectiveContractError(
                "objective_surface_missing",
                "objective profile requires unique surface roles",
            )
        try:
            targets = tuple(
                sorted(unique_strings(self.target_requirements, "target_requirements"))
            )
        except (TypeError, ValueError) as error:
            raise ObjectiveContractError(
                "objective_target_invalid",
                "objective target requirements must be unique stable strings",
            ) from error
        if not targets:
            raise ObjectiveContractError(
                "objective_target_invalid",
                "objective profile requires target requirements",
            )
        try:
            metrics = tuple(sorted(unique_strings(self.metric_names, "metric_names")))
        except (TypeError, ValueError) as error:
            raise ObjectiveContractError(
                "objective_metric_invalid",
                "objective metric names must be unique stable strings",
            ) from error
        if not metrics:
            raise ObjectiveContractError(
                "objective_metric_invalid", "objective profile requires metric names"
            )
        _stable_id(self.metric_schema_id, "metric_schema_id")
        object.__setattr__(self, "supported_execution_capabilities", capabilities)
        object.__setattr__(self, "required_surface_roles", roles)
        object.__setattr__(self, "target_requirements", targets)
        object.__setattr__(self, "metric_names", metrics)
        try:
            non_claims = tuple(sorted(unique_strings(self.non_claims, "non_claims")))
        except (TypeError, ValueError) as error:
            raise ObjectiveContractError(
                "objective_config_invalid",
                "objective non-claims must be unique stable strings",
            ) from error
        object.__setattr__(self, "non_claims", non_claims)

    def supports(self, capability: str) -> bool:
        return capability in self.supported_execution_capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "capability_schema_version": self.capability_schema_version,
            "supported_execution_capabilities": list(
                self.supported_execution_capabilities
            ),
            "required_surface_roles": list(self.required_surface_roles),
            "target_requirements": list(self.target_requirements),
            "metric_schema_id": self.metric_schema_id,
            "metric_names": list(self.metric_names),
            "non_claims": list(self.non_claims),
        }

    @property
    def digest(self) -> str:
        return objective_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveCapabilityProfile:
        _strict_fields(
            payload,
            {
                "identity",
                "capability_schema_version",
                "supported_execution_capabilities",
                "required_surface_roles",
                "target_requirements",
                "metric_schema_id",
                "metric_names",
                "non_claims",
            },
            "capability profile",
        )
        return cls(
            identity=ObjectiveIdentity.from_dict(
                mapping(payload["identity"], "identity")
            ),
            capability_schema_version=str(payload["capability_schema_version"]),
            supported_execution_capabilities=tuple(
                payload["supported_execution_capabilities"]
            ),
            required_surface_roles=tuple(payload["required_surface_roles"]),
            target_requirements=tuple(payload["target_requirements"]),
            metric_schema_id=str(payload["metric_schema_id"]),
            metric_names=tuple(payload["metric_names"]),
            non_claims=tuple(payload["non_claims"]),
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    """Objective-owned configuration; architecture selection is deliberately absent."""

    identity: ObjectiveIdentity
    values: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    config_schema_version: str = OBJECTIVE_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ObjectiveIdentity):
            raise ObjectiveContractError(
                "objective_config_invalid",
                "objective config requires ObjectiveIdentity",
            )
        if self.config_schema_version != OBJECTIVE_CONFIG_SCHEMA_VERSION:
            raise ObjectiveContractError(
                "objective_config_invalid", "unsupported objective config schema"
            )
        object.__setattr__(self, "values", freeze_json_mapping(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "config_schema_version": self.config_schema_version,
            "values": json_value(self.values),
        }

    @property
    def digest(self) -> str:
        return objective_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveConfig:
        _strict_fields(
            payload,
            {"identity", "config_schema_version", "values"},
            "objective config",
        )
        return cls(
            identity=ObjectiveIdentity.from_dict(
                mapping(payload["identity"], "identity")
            ),
            config_schema_version=str(payload["config_schema_version"]),
            values=mapping(payload["values"], "values"),
        )


@dataclass(frozen=True)
class ObjectiveExecutionDescriptor:
    identity: ObjectiveIdentity
    capability_profile_digest: str
    config_digest: str
    resolved_surface_identity: str
    metric_schema_id: str
    implementation_identity: str
    descriptor_schema_version: str = OBJECTIVE_EXECUTION_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ObjectiveIdentity):
            raise ObjectiveContractError(
                "objective_config_invalid",
                "objective descriptor requires ObjectiveIdentity",
            )
        if (
            self.descriptor_schema_version
            != OBJECTIVE_EXECUTION_DESCRIPTOR_SCHEMA_VERSION
        ):
            raise ObjectiveContractError(
                "objective_config_invalid", "unsupported objective descriptor schema"
            )
        for name in (
            "capability_profile_digest",
            "config_digest",
            "resolved_surface_identity",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _stable_id(self.metric_schema_id, "metric_schema_id")
        _stable_id(self.implementation_identity, "implementation_identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "descriptor_schema_version": self.descriptor_schema_version,
            "capability_profile_digest": self.capability_profile_digest,
            "config_digest": self.config_digest,
            "resolved_surface_identity": self.resolved_surface_identity,
            "metric_schema_id": self.metric_schema_id,
            "implementation_identity": self.implementation_identity,
        }

    @property
    def digest(self) -> str:
        return objective_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveExecutionDescriptor:
        _strict_fields(
            payload,
            {
                "identity",
                "descriptor_schema_version",
                "capability_profile_digest",
                "config_digest",
                "resolved_surface_identity",
                "metric_schema_id",
                "implementation_identity",
            },
            "objective execution descriptor",
        )
        return cls(
            identity=ObjectiveIdentity.from_dict(
                mapping(payload["identity"], "identity")
            ),
            descriptor_schema_version=str(payload["descriptor_schema_version"]),
            capability_profile_digest=str(payload["capability_profile_digest"]),
            config_digest=str(payload["config_digest"]),
            resolved_surface_identity=str(payload["resolved_surface_identity"]),
            metric_schema_id=str(payload["metric_schema_id"]),
            implementation_identity=str(payload["implementation_identity"]),
        )


__all__ = [
    "OBJECTIVE_CAPABILITY_SCHEMA_VERSION",
    "OBJECTIVE_CONFIG_SCHEMA_VERSION",
    "OBJECTIVE_EXECUTION_DESCRIPTOR_SCHEMA_VERSION",
    "ObjectiveCapabilityProfile",
    "ObjectiveConfig",
    "ObjectiveContractError",
    "ObjectiveExecutionDescriptor",
    "ObjectiveIdentity",
    "ResolvedObjectiveSelection",
    "canonical_objective_json",
    "objective_digest",
]
