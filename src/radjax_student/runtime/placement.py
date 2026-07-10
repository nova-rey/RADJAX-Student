"""Portable placement declarations with no topology or backend realization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.runtime.errors import RuntimeIssue
from radjax_student.runtime.models import (
    PLACEMENT_POLICIES,
    RuntimeConfig,
    freeze_json_mapping,
    json_value,
)

PlacementIntent = Literal[
    "single_device",
    "replicated",
    "data_sharded",
    "model_sharded",
    "automatic",
    "unspecified",
]
LogicalAxisRole = Literal[
    "replicated",
    "data",
    "model",
    "automatic",
    "unspecified",
]
PlacementResolutionStatus = Literal["unresolved", "fail"]

PLACEMENT_INTENTS: tuple[str, ...] = PLACEMENT_POLICIES
LOGICAL_AXIS_ROLES: tuple[str, ...] = (
    "replicated",
    "data",
    "model",
    "automatic",
    "unspecified",
)
PLACEMENT_CAPABILITY_MAPPING_VERSION = "placement_capabilities.v1"
PLACEMENT_CAPABILITY_MAPPING: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "single_device": ("placement.single_device_v1",),
        "replicated": ("placement.replicated_v1",),
        "data_sharded": ("placement.data_sharded_v1",),
        "model_sharded": ("placement.model_sharded_v1",),
        "automatic": (),
        "unspecified": (),
    }
)
PLACEMENT_BLOCKER_CODES: tuple[str, ...] = (
    "placement_value_path_duplicate",
    "placement_axis_duplicate",
    "placement_axis_unknown",
    "placement_axis_size_invalid",
    "placement_intent_invalid",
    "placement_constraint_conflict",
    "placement_capability_missing",
    "placement_plan_invalid",
    "placement_internal_error",
)
PLACEMENT_WARNING_CODES: tuple[str, ...] = (
    "placement_automatic_unresolved",
    "placement_unspecified",
    "placement_axis_size_unknown",
    "placement_plugin_defined_axis",
    "placement_constraint_not_evaluated",
    "placement_declaration_not_execution_proof",
)
COMMON_LOGICAL_AXIS_NAMES: tuple[str, ...] = (
    "batch",
    "sequence",
    "vocab",
    "model",
    "heads",
    "layers",
    "expert",
    "state",
)
PARTITIONED_AXIS_CONSTRAINTS: tuple[str, ...] = (
    "must_be_divisible_by_device_count",
    "must_not_cross_process",
)
_AXIS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class PlacementContractError(ValueError):
    """Structured invalid placement declaration with a stable code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        if code not in PLACEMENT_BLOCKER_CODES:
            raise ValueError(f"unknown placement blocker code: {code}")
        self.issue = RuntimeIssue.create(code, message, **details)
        super().__init__(f"{code}: {message}")

    @property
    def code(self) -> str:
        return self.issue.code

    def to_dict(self) -> dict[str, Any]:
        return self.issue.to_dict()


@dataclass(frozen=True)
class LogicalAxisSpec:
    """A semantic, topology-free logical axis declared by a plugin or caller."""

    name: str
    size: int | None = None
    sharding_role: LogicalAxisRole = "unspecified"
    required: bool = False
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        _require_axis_name(self.name)
        if self.size is not None and (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise PlacementContractError(
                "placement_axis_size_invalid",
                "logical axis size must be positive when declared",
                axis_name=self.name,
                size=self.size,
            )
        if self.sharding_role not in LOGICAL_AXIS_ROLES:
            raise PlacementContractError(
                "placement_intent_invalid",
                "logical axis has an unsupported sharding role",
                axis_name=self.name,
                sharding_role=self.sharding_role,
            )
        if not isinstance(self.required, bool):
            raise TypeError("logical axis required must be boolean")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("logical axis metadata must be a mapping")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "sharding_role": self.sharding_role,
            "required": self.required,
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LogicalAxisSpec:
        return cls(
            name=_string(payload["name"], "name"),
            size=_optional_positive_int(payload.get("size"), "size"),
            sharding_role=_string(
                payload.get("sharding_role", "unspecified"), "sharding_role"
            ),
            required=_bool(payload.get("required", False), "required"),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
        )


LogicalAxis = LogicalAxisSpec


@dataclass(frozen=True)
class ValuePlacementSpec:
    """Placement declaration for one stable logical value path."""

    value_path: str
    placement: PlacementIntent = "unspecified"
    logical_axes: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        _require_value_path(self.value_path)
        _require_placement_intent(self.placement, "placement")
        axes = _unique_strings(self.logical_axes, "logical_axes")
        capabilities = _sorted_strings(
            (*self.required_capabilities, *placement_capabilities(self.placement)),
            "required_capabilities",
        )
        constraints = _unique_sorted_strings(self.constraints, "constraints")
        if self.placement == "unspecified" and (capabilities or constraints):
            raise PlacementContractError(
                "placement_constraint_conflict",
                "unspecified placement cannot carry concrete capabilities "
                "or constraints",
                value_path=self.value_path,
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("value placement metadata must be a mapping")
        object.__setattr__(self, "logical_axes", axes)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_path": self.value_path,
            "placement": self.placement,
            "logical_axes": list(self.logical_axes),
            "required_capabilities": list(self.required_capabilities),
            "constraints": list(self.constraints),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValuePlacementSpec:
        return cls(
            value_path=_string(payload["value_path"], "value_path"),
            placement=_string(payload.get("placement", "unspecified"), "placement"),
            logical_axes=_strings(payload.get("logical_axes", ()), "logical_axes"),
            required_capabilities=_strings(
                payload.get("required_capabilities", ()), "required_capabilities"
            ),
            constraints=_strings(payload.get("constraints", ()), "constraints"),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class PlacementPlan:
    """Immutable placement intent over logical value paths, never a device plan."""

    plan_id: str
    values: tuple[ValuePlacementSpec, ...]
    logical_axis_catalog: tuple[LogicalAxisSpec, ...] = ()
    default_placement: PlacementIntent = "unspecified"
    required_capabilities: tuple[str, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = (
        "placement_not_resolved_to_devices",
        "mesh_not_created",
        "sharding_not_executed",
    )
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise PlacementContractError(
                "placement_plan_invalid", "plan_id must be a nonempty string"
            )
        values = tuple(self.values)
        if any(not isinstance(item, ValuePlacementSpec) for item in values):
            raise TypeError("values must contain ValuePlacementSpec values")
        _require_placement_intent(self.default_placement, "default_placement")
        catalog = tuple(self.logical_axis_catalog)
        if any(not isinstance(item, LogicalAxisSpec) for item in catalog):
            raise TypeError("logical_axis_catalog must contain LogicalAxisSpec values")
        _validate_plan(values, catalog)
        capabilities = _sorted_strings(
            (
                *self.required_capabilities,
                *placement_capabilities(self.default_placement),
                *(
                    capability
                    for value in values
                    for capability in value.required_capabilities
                ),
            ),
            "required_capabilities",
        )
        warnings = _placement_warnings(
            values, catalog, self.default_placement, self.warnings
        )
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("placement plan metadata must be a mapping")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "logical_axis_catalog", catalog)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_not_made", claims)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def value(self, value_path: str) -> ValuePlacementSpec:
        for value in self.values:
            if value.value_path == value_path:
                return value
        raise KeyError(f"unknown placement value path: {value_path}")

    def effective_placement(
        self,
        value_path: str,
        config: RuntimeConfig | None = None,
    ) -> PlacementIntent:
        return effective_placement(self.value(value_path), self, config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "values": [item.to_dict() for item in self.values],
            "logical_axis_catalog": [
                item.to_dict() for item in self.logical_axis_catalog
            ],
            "default_placement": self.default_placement,
            "required_capabilities": list(self.required_capabilities),
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlacementPlan:
        return cls(
            plan_id=_string(payload["plan_id"], "plan_id"),
            values=tuple(
                ValuePlacementSpec.from_dict(_mapping(item, "value"))
                for item in _sequence(payload.get("values", ()), "values")
            ),
            logical_axis_catalog=tuple(
                LogicalAxisSpec.from_dict(_mapping(item, "logical axis"))
                for item in _sequence(
                    payload.get("logical_axis_catalog", ()), "logical_axis_catalog"
                )
            ),
            default_placement=_string(
                payload.get("default_placement", "unspecified"), "default_placement"
            ),
            required_capabilities=_strings(
                payload.get("required_capabilities", ()), "required_capabilities"
            ),
            warnings=_issues_from_payload(payload.get("warnings", ()), "warnings"),
            claims_not_made=_strings(
                payload.get("claims_not_made", ()), "claims_not_made"
            ),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class PlacementResolution:
    """Reserved, serializable result for later backend translation of intent."""

    status: PlacementResolutionStatus
    intent: PlacementIntent
    resolved_backend: str | None = None
    resolved_devices: tuple[str, ...] = ()
    resolved_sharding: Mapping[str, Any] = MappingProxyType({})
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = (
        "device_resolution_not_implemented",
        "concrete_sharding_not_implemented",
    )

    def __post_init__(self) -> None:
        if self.status not in ("unresolved", "fail"):
            raise ValueError("placement resolution status must be unresolved or fail")
        _require_placement_intent(self.intent, "intent")
        if self.resolved_backend is not None and (
            not isinstance(self.resolved_backend, str) or not self.resolved_backend
        ):
            raise ValueError("resolved_backend must be a nonempty string when set")
        devices = _unique_strings(self.resolved_devices, "resolved_devices")
        if devices:
            raise PlacementContractError(
                "placement_plan_invalid",
                "P2.6 resolution must not name concrete devices",
            )
        if not isinstance(self.resolved_sharding, Mapping):
            raise TypeError("resolved_sharding must be a mapping")
        if self.resolved_sharding:
            raise PlacementContractError(
                "placement_plan_invalid",
                "P2.6 resolution must not contain concrete sharding objects",
            )
        blockers = _issues(self.blockers, "blockers", PLACEMENT_BLOCKER_CODES)
        warnings = _issues(self.warnings, "warnings", PLACEMENT_WARNING_CODES)
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        if self.status == "unresolved" and blockers:
            raise ValueError("unresolved placement resolution cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failed placement resolution must contain blockers")
        object.__setattr__(self, "resolved_devices", devices)
        object.__setattr__(
            self, "resolved_sharding", freeze_json_mapping(self.resolved_sharding)
        )
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_not_made", claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "intent": self.intent,
            "resolved_backend": self.resolved_backend,
            "resolved_devices": list(self.resolved_devices),
            "resolved_sharding": json_value(self.resolved_sharding),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlacementResolution:
        return cls(
            status=_string(payload["status"], "status"),
            intent=_string(payload["intent"], "intent"),
            resolved_backend=_optional_string(payload.get("resolved_backend")),
            resolved_devices=_strings(
                payload.get("resolved_devices", ()), "resolved_devices"
            ),
            resolved_sharding=_mapping(
                payload.get("resolved_sharding", {}), "resolved_sharding"
            ),
            blockers=_issues_from_payload(payload.get("blockers", ()), "blockers"),
            warnings=_issues_from_payload(payload.get("warnings", ()), "warnings"),
            claims_not_made=_strings(
                payload.get("claims_not_made", ()), "claims_not_made"
            ),
        )


def placement_capabilities(intent: PlacementIntent) -> tuple[str, ...]:
    """Return the versioned capability declaration implied by one intent."""

    _require_placement_intent(intent, "intent")
    return PLACEMENT_CAPABILITY_MAPPING[intent]


def effective_placement(
    value: ValuePlacementSpec,
    plan: PlacementPlan,
    config: RuntimeConfig | None = None,
) -> PlacementIntent:
    """Apply value -> plan -> runtime-config -> unresolved precedence only."""

    if value.placement != "unspecified":
        return value.placement
    if plan.default_placement != "unspecified":
        return plan.default_placement
    if config is not None and config.placement_policy != "unspecified":
        return config.placement_policy
    return "unspecified"


def unresolved_placement_resolution(
    intent: PlacementIntent,
) -> PlacementResolution:
    """Represent declared intent while reserving all concrete realization."""

    warning_code = (
        "placement_automatic_unresolved"
        if intent == "automatic"
        else "placement_unspecified"
        if intent == "unspecified"
        else "placement_declaration_not_execution_proof"
    )
    return PlacementResolution(
        status="unresolved",
        intent=intent,
        warnings=(
            RuntimeIssue.create(
                warning_code,
                "placement intent remains unresolved without topology translation",
                intent=intent,
            ),
        ),
    )


def _validate_plan(
    values: tuple[ValuePlacementSpec, ...],
    catalog: tuple[LogicalAxisSpec, ...],
) -> None:
    paths = [value.value_path for value in values]
    duplicate_paths = _duplicates(paths)
    if duplicate_paths:
        raise PlacementContractError(
            "placement_value_path_duplicate",
            "placement plan contains duplicate value paths",
            value_paths=duplicate_paths,
        )
    names = [axis.name for axis in catalog]
    duplicate_axes = _duplicates(names)
    if duplicate_axes:
        raise PlacementContractError(
            "placement_axis_duplicate",
            "logical axis catalog contains duplicate names",
            axis_names=duplicate_axes,
        )
    catalog_by_name = {axis.name: axis for axis in catalog}
    for value in values:
        unknown = tuple(
            axis for axis in value.logical_axes if axis not in catalog_by_name
        )
        if unknown:
            raise PlacementContractError(
                "placement_axis_unknown",
                "value placement references an undeclared logical axis",
                value_path=value.value_path,
                axis_names=unknown,
            )
        roles = {catalog_by_name[axis].sharding_role for axis in value.logical_axes}
        if value.placement == "data_sharded" and "data" not in roles:
            raise PlacementContractError(
                "placement_constraint_conflict",
                "data-sharded placement requires a declared data logical axis",
                value_path=value.value_path,
            )
        if value.placement == "model_sharded" and "model" not in roles:
            raise PlacementContractError(
                "placement_constraint_conflict",
                "model-sharded placement requires a declared model logical axis",
                value_path=value.value_path,
            )
        if value.placement == "replicated" and set(value.constraints) & set(
            PARTITIONED_AXIS_CONSTRAINTS
        ):
            raise PlacementContractError(
                "placement_constraint_conflict",
                "replicated placement cannot carry partitioned-axis constraints",
                value_path=value.value_path,
            )


def _placement_warnings(
    values: tuple[ValuePlacementSpec, ...],
    catalog: tuple[LogicalAxisSpec, ...],
    default_placement: PlacementIntent,
    supplied: tuple[RuntimeIssue, ...],
) -> tuple[RuntimeIssue, ...]:
    warnings = list(_issues(supplied, "warnings", PLACEMENT_WARNING_CODES))
    for axis in catalog:
        if axis.size is None:
            warnings.append(
                RuntimeIssue.create(
                    "placement_axis_size_unknown",
                    "logical axis size is intentionally unknown",
                    axis_name=axis.name,
                )
            )
        if axis.name not in COMMON_LOGICAL_AXIS_NAMES:
            warnings.append(
                RuntimeIssue.create(
                    "placement_plugin_defined_axis",
                    "placement plan preserves a plugin-defined logical axis",
                    axis_name=axis.name,
                )
            )
    intents = (default_placement, *(item.placement for item in values))
    if "automatic" in intents:
        warnings.append(
            RuntimeIssue.create(
                "placement_automatic_unresolved",
                "automatic placement is declared but not resolved in P2.6",
            )
        )
    if "unspecified" in intents:
        warnings.append(
            RuntimeIssue.create(
                "placement_unspecified",
                "unspecified placement remains distinct from automatic placement",
            )
        )
    if any(item.constraints for item in values):
        warnings.append(
            RuntimeIssue.create(
                "placement_constraint_not_evaluated",
                "placement constraints are declarations and were not "
                "topology-evaluated",
            )
        )
    if any(item.placement not in ("automatic", "unspecified") for item in values):
        warnings.append(
            RuntimeIssue.create(
                "placement_declaration_not_execution_proof",
                "placement declarations do not prove concrete device realization",
            )
        )
    return _deduplicate_issues(warnings)


def _require_axis_name(name: object) -> None:
    if not isinstance(name, str) or not _AXIS_NAME.fullmatch(name):
        raise PlacementContractError(
            "placement_intent_invalid",
            "logical axis name must be backend-neutral and deterministic",
            axis_name=name if isinstance(name, str) else None,
        )


def _require_value_path(path: object) -> None:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith(".")
        or path.endswith(".")
    ):
        raise PlacementContractError(
            "placement_intent_invalid",
            "value_path must be a stable nonempty logical path",
        )


def _require_placement_intent(intent: object, name: str) -> None:
    if intent not in PLACEMENT_INTENTS:
        raise PlacementContractError(
            "placement_intent_invalid",
            f"{name} must be one of the public placement intents",
            observed_intent=intent if isinstance(intent, str) else None,
        )


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlacementContractError(
            "placement_axis_size_invalid",
            f"{name} must be positive when declared",
            size=value if isinstance(value, int) else None,
        )
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value, "value")


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _unique_strings(value: object, name: str) -> tuple[str, ...]:
    result = _strings(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _unique_sorted_strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(sorted(set(_unique_strings(value, name))))


def _sorted_strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(sorted(set(_strings(value, name))))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(value)


def _issues(
    value: object,
    name: str,
    valid_codes: tuple[str, ...],
) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, RuntimeIssue) for item in result):
        raise TypeError(f"{name} must contain RuntimeIssue values")
    invalid = [item.code for item in result if item.code not in valid_codes]
    if invalid:
        raise ValueError(f"{name} contains unknown placement codes: {invalid}")
    return result


def _issues_from_payload(value: object, name: str) -> tuple[RuntimeIssue, ...]:
    return tuple(
        RuntimeIssue.from_dict(_mapping(item, name)) for item in _sequence(value, name)
    )


def _duplicates(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if values.count(value) > 1}))


def _deduplicate_issues(issues: list[RuntimeIssue]) -> tuple[RuntimeIssue, ...]:
    result: list[RuntimeIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        identity = (issue.code, issue.message, repr(sorted(issue.details.items())))
        if identity not in seen:
            result.append(issue)
            seen.add(identity)
    return tuple(result)
