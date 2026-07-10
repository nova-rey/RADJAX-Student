"""Pure, deterministic runtime backend selection over declared facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from radjax_student.runtime.errors import RUNTIME_ERROR_CODES, RuntimeIssue
from radjax_student.runtime.inspection import RuntimeInspection
from radjax_student.runtime.models import RuntimeConfig
from radjax_student.runtime.registry import (
    RuntimeBackendDescriptor,
    RuntimeBackendRegistry,
)

SelectionStatus = Literal["pass", "fail"]
RUNTIME_SELECTION_WARNING_CODES: tuple[str, ...] = (
    "runtime_compatible_fallback_used",
    "runtime_capability_declared_not_proven",
    "runtime_platform_inferred",
    "runtime_precision_unevaluated",
    "runtime_distributed_unevaluated",
    "runtime_selection_used_tiebreak",
)
RUNTIME_SELECTION_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "backend_not_initialized",
    "array_creation_not_tested",
    "array_placement_not_tested",
    "compilation_not_tested",
    "synchronization_not_tested",
    "execution_not_tested",
    "precision_behavior_not_proven",
    "distributed_execution_not_tested",
)
AUTOMATIC_PLATFORM_PREFERENCE: tuple[str, ...] = ("gpu", "tpu", "metal", "cpu")


@dataclass(frozen=True)
class RuntimeSelectionResult:
    """Selection decision and evidence; it never retains a backend object."""

    status: SelectionStatus
    requested_config: RuntimeConfig
    selected_backend: RuntimeBackendDescriptor | None
    selected_platform: str | None
    considered_backends: tuple[RuntimeBackendDescriptor, ...]
    required_capabilities: tuple[str, ...]
    satisfied_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    fallback_used: bool
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = RUNTIME_SELECTION_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("runtime selection status must be pass or fail")
        if not isinstance(self.requested_config, RuntimeConfig):
            raise TypeError("requested_config must be RuntimeConfig")
        if self.selected_backend is not None and not isinstance(
            self.selected_backend, RuntimeBackendDescriptor
        ):
            raise TypeError("selected_backend must be RuntimeBackendDescriptor")
        if self.selected_platform is not None and (
            not isinstance(self.selected_platform, str) or not self.selected_platform
        ):
            raise ValueError("selected_platform must be a nonempty string when set")
        considered = tuple(self.considered_backends)
        if any(not isinstance(item, RuntimeBackendDescriptor) for item in considered):
            raise TypeError("considered_backends must contain RuntimeBackendDescriptor")
        required = _unique_strings(self.required_capabilities, "required_capabilities")
        satisfied = _unique_strings(
            self.satisfied_capabilities, "satisfied_capabilities"
        )
        missing = _unique_strings(self.missing_capabilities, "missing_capabilities")
        if set(satisfied) & set(missing):
            raise ValueError("satisfied and missing capabilities must not overlap")
        if set(required) != set(satisfied) | set(missing):
            raise ValueError("required capabilities must be satisfied or missing")
        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used must be boolean")
        blockers = _issues(self.blockers, "blockers")
        warnings = _issues(self.warnings, "warnings")
        unknown_blockers = [
            item.code for item in blockers if item.code not in RUNTIME_ERROR_CODES
        ]
        if unknown_blockers:
            raise ValueError(
                "selection blockers have unknown codes: " + ", ".join(unknown_blockers)
            )
        unknown_warnings = [
            item.code
            for item in warnings
            if item.code not in RUNTIME_SELECTION_WARNING_CODES
        ]
        if unknown_warnings:
            raise ValueError(
                "selection warnings have unknown codes: " + ", ".join(unknown_warnings)
            )
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        if self.status == "pass" and blockers:
            raise ValueError("passing selection cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing selection must contain blockers")
        if self.status == "pass" and self.selected_backend is None:
            raise ValueError("passing selection must identify a backend")
        object.__setattr__(self, "considered_backends", considered)
        object.__setattr__(self, "required_capabilities", required)
        object.__setattr__(self, "satisfied_capabilities", satisfied)
        object.__setattr__(self, "missing_capabilities", missing)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_not_made", claims)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "requested_config": self.requested_config.to_dict(),
            "selected_backend": (
                None
                if self.selected_backend is None
                else self.selected_backend.to_dict()
            ),
            "selected_platform": self.selected_platform,
            "considered_backends": [
                item.to_dict() for item in self.considered_backends
            ],
            "required_capabilities": list(self.required_capabilities),
            "satisfied_capabilities": list(self.satisfied_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "fallback_used": self.fallback_used,
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeSelectionResult:
        selected = payload.get("selected_backend")
        considered = payload.get("considered_backends", ())
        return cls(
            status=str(payload["status"]),
            requested_config=RuntimeConfig.from_dict(
                _mapping(payload["requested_config"], "requested_config")
            ),
            selected_backend=(
                None
                if selected is None
                else RuntimeBackendDescriptor.from_dict(
                    _mapping(selected, "selected_backend")
                )
            ),
            selected_platform=(
                None
                if payload.get("selected_platform") is None
                else str(payload["selected_platform"])
            ),
            considered_backends=tuple(
                RuntimeBackendDescriptor.from_dict(_mapping(item, "considered_backend"))
                for item in _sequence(considered, "considered_backends")
            ),
            required_capabilities=_strings(
                payload.get("required_capabilities", ()), "required_capabilities"
            ),
            satisfied_capabilities=_strings(
                payload.get("satisfied_capabilities", ()), "satisfied_capabilities"
            ),
            missing_capabilities=_strings(
                payload.get("missing_capabilities", ()), "missing_capabilities"
            ),
            fallback_used=_bool(payload.get("fallback_used"), "fallback_used"),
            blockers=_issues_from_payload(payload.get("blockers", ()), "blockers"),
            warnings=_issues_from_payload(payload.get("warnings", ()), "warnings"),
            claims_not_made=_strings(
                payload.get("claims_not_made", ()), "claims_not_made"
            ),
        )


@dataclass(frozen=True)
class _Candidate:
    descriptor: RuntimeBackendDescriptor
    selected_platform: str | None
    fallback_used: bool
    blockers: tuple[RuntimeIssue, ...]

    @property
    def eligible(self) -> bool:
        return not self.blockers


def select_runtime_backend(
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    registry: RuntimeBackendRegistry,
) -> RuntimeSelectionResult:
    """Select from declared candidates using supplied inspection facts only."""

    try:
        descriptors = registry.describe(inspection)
        return _select(config, inspection, descriptors)
    except Exception as exc:
        return RuntimeSelectionResult(
            status="fail",
            requested_config=config,
            selected_backend=None,
            selected_platform=None,
            considered_backends=(),
            required_capabilities=config.required_capabilities,
            satisfied_capabilities=(),
            missing_capabilities=config.required_capabilities,
            fallback_used=False,
            blockers=(
                RuntimeIssue.create(
                    "runtime_selection_internal_error",
                    "runtime selection could not produce a coherent decision",
                    exception_type=type(exc).__name__,
                ),
            ),
        )


def _select(
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    descriptors: tuple[RuntimeBackendDescriptor, ...],
) -> RuntimeSelectionResult:
    descriptor_by_id = {item.backend_id: item for item in descriptors}
    visible_platforms = _visible_platforms(inspection)
    requested_platform = config.platform_preference
    if config.backend_id is not None and config.backend_id not in descriptor_by_id:
        if config.fallback_policy == "disallowed":
            return _failure(
                config,
                descriptors,
                blockers=(
                    RuntimeIssue.create(
                        "runtime_backend_not_found",
                        "requested runtime backend is not registered",
                        backend_id=config.backend_id,
                        registered_backend_ids=tuple(descriptor_by_id),
                    ),
                ),
            )
        candidate_descriptors = descriptors
    elif config.backend_id is not None:
        candidate_descriptors = (descriptor_by_id[config.backend_id],)
    else:
        candidate_descriptors = descriptors

    candidates = tuple(
        _evaluate_candidate(
            descriptor,
            config,
            visible_platforms,
            requested_platform,
            backend_fallback=(
                config.backend_id is not None
                and descriptor.backend_id != config.backend_id
            ),
        )
        for descriptor in candidate_descriptors
    )
    eligible = tuple(item for item in candidates if item.eligible)
    if not eligible:
        blockers = _best_failure_blockers(candidates)
        return _failure(config, descriptors, blockers=blockers)

    ranked = tuple(sorted(eligible, key=lambda item: _candidate_key(item, config)))
    selected = ranked[0]
    warnings = _selection_warnings(config, selected, ranked)
    return RuntimeSelectionResult(
        status="pass",
        requested_config=config,
        selected_backend=selected.descriptor,
        selected_platform=selected.selected_platform,
        considered_backends=descriptors,
        required_capabilities=config.required_capabilities,
        satisfied_capabilities=config.required_capabilities,
        missing_capabilities=(),
        fallback_used=selected.fallback_used,
        warnings=warnings,
    )


def _evaluate_candidate(
    descriptor: RuntimeBackendDescriptor,
    config: RuntimeConfig,
    visible_platforms: tuple[str, ...],
    requested_platform: str,
    *,
    backend_fallback: bool,
) -> _Candidate:
    blockers: list[RuntimeIssue] = []
    fallback_used = backend_fallback
    selected_platform: str | None = None
    if descriptor.availability.status != "available":
        blockers.extend(descriptor.availability.reasons)
    if requested_platform in ("automatic", "unspecified"):
        if requested_platform == "automatic":
            selected_platform = _automatic_platform(
                descriptor.supported_platforms,
                visible_platforms,
            )
            if selected_platform is None:
                blockers.append(
                    _platform_unavailable(
                        descriptor, requested_platform, visible_platforms
                    )
                )
    elif requested_platform not in descriptor.supported_platforms:
        blockers.append(
            _platform_unavailable(descriptor, requested_platform, visible_platforms)
        )
    elif requested_platform in visible_platforms:
        selected_platform = requested_platform
    else:
        fallback_platform = _automatic_platform(
            descriptor.supported_platforms,
            visible_platforms,
        )
        if (
            fallback_platform is not None
            and config.fallback_policy == "allow_compatible"
        ):
            selected_platform = fallback_platform
            fallback_used = True
        else:
            blockers.append(
                _platform_unavailable(descriptor, requested_platform, visible_platforms)
            )
            if fallback_platform is not None:
                blockers.append(
                    RuntimeIssue.create(
                        "runtime_fallback_disallowed",
                        "a compatible platform fallback exists but policy disallows it",
                        backend_id=descriptor.backend_id,
                        requested_platform=requested_platform,
                        compatible_platform=fallback_platform,
                    )
                )
    profile = descriptor.capability_profile
    for capability in config.required_capabilities:
        if capability not in profile.capabilities:
            blockers.append(
                RuntimeIssue.create(
                    "runtime_capability_missing",
                    "runtime backend does not declare a required capability",
                    backend_id=descriptor.backend_id,
                    capability=capability,
                )
            )
    blockers.extend(_policy_blockers(descriptor, config))
    return _Candidate(
        descriptor=descriptor,
        selected_platform=selected_platform,
        fallback_used=fallback_used,
        blockers=tuple(blockers),
    )


def _policy_blockers(
    descriptor: RuntimeBackendDescriptor,
    config: RuntimeConfig,
) -> tuple[RuntimeIssue, ...]:
    requirements: list[tuple[str, str]] = []
    if config.distributed_policy == "required":
        requirements.append(("distributed_policy", "runtime.multi_process_v1"))
    if config.placement_policy not in ("automatic", "unspecified", "single_device"):
        requirements.append(
            ("placement_policy", f"placement.{config.placement_policy}_v1")
        )
    if config.compilation_policy == "jit":
        requirements.append(("compilation_policy", "compilation.jit_v1"))
    blockers = [
        RuntimeIssue.create(
            "runtime_policy_unsupported",
            "runtime backend declaration cannot satisfy a requested policy",
            backend_id=descriptor.backend_id,
            policy=policy,
            required_capability=capability,
        )
        for policy, capability in requirements
        if capability not in descriptor.capability_profile.capabilities
    ]
    return tuple(blockers)


def _best_failure_blockers(
    candidates: tuple[_Candidate, ...],
) -> tuple[RuntimeIssue, ...]:
    if not candidates:
        return (
            RuntimeIssue.create(
                "runtime_backend_not_found",
                "no runtime backends are registered",
            ),
        )
    selected = min(
        candidates, key=lambda item: (len(item.blockers), item.descriptor.backend_id)
    )
    return selected.blockers


def _selection_warnings(
    config: RuntimeConfig,
    selected: _Candidate,
    ranked: tuple[_Candidate, ...],
) -> tuple[RuntimeIssue, ...]:
    warnings: list[RuntimeIssue] = []
    if selected.fallback_used:
        warnings.append(
            RuntimeIssue.create(
                "runtime_compatible_fallback_used",
                "selection used an explicitly permitted compatible fallback",
                requested_backend_id=config.backend_id,
                selected_backend_id=selected.descriptor.backend_id,
                requested_platform=config.platform_preference,
                selected_platform=selected.selected_platform,
            )
        )
    if config.platform_preference == "automatic":
        warnings.append(
            RuntimeIssue.create(
                "runtime_platform_inferred",
                "automatic platform selection used the documented preference order",
                selected_platform=selected.selected_platform,
                preference_order=AUTOMATIC_PLATFORM_PREFERENCE,
            )
        )
    if config.precision_policy not in ("automatic", "unspecified"):
        warnings.append(
            RuntimeIssue.create(
                "runtime_precision_unevaluated",
                "selection does not prove precision behavior before execution",
                requested_precision=config.precision_policy,
            )
        )
    if config.distributed_policy == "auto":
        warnings.append(
            RuntimeIssue.create(
                "runtime_distributed_unevaluated",
                "selection does not infer distributed execution from declarations",
            )
        )
    if selected.descriptor.capability_profile.capabilities:
        warnings.append(
            RuntimeIssue.create(
                "runtime_capability_declared_not_proven",
                "backend capabilities are declarations and not execution proof",
                backend_id=selected.descriptor.backend_id,
            )
        )
    if len(ranked) > 1:
        warnings.append(
            RuntimeIssue.create(
                "runtime_selection_used_tiebreak",
                "multiple backends were eligible; deterministic ordering selected one",
                eligible_backend_ids=tuple(
                    item.descriptor.backend_id for item in ranked
                ),
                selected_backend_id=selected.descriptor.backend_id,
            )
        )
    return tuple(warnings)


def _candidate_key(
    candidate: _Candidate, config: RuntimeConfig
) -> tuple[int, int, str]:
    exact_platform = candidate.selected_platform == config.platform_preference
    return (
        0 if exact_platform else 1,
        0 if not candidate.fallback_used else 1,
        candidate.descriptor.backend_id,
    )


def _visible_platforms(inspection: RuntimeInspection) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.platform
                for item in inspection.device_inventory.devices
                if item.platform is not None
            }
        )
    )


def _automatic_platform(
    supported_platforms: tuple[str, ...],
    visible_platforms: tuple[str, ...],
) -> str | None:
    compatible = set(supported_platforms) & set(visible_platforms)
    return next(
        (item for item in AUTOMATIC_PLATFORM_PREFERENCE if item in compatible), None
    )


def _platform_unavailable(
    descriptor: RuntimeBackendDescriptor,
    requested_platform: str,
    visible_platforms: tuple[str, ...],
) -> RuntimeIssue:
    return RuntimeIssue.create(
        "requested_platform_unavailable",
        "requested platform is not a compatible visible backend target",
        backend_id=descriptor.backend_id,
        requested_platform=requested_platform,
        supported_platforms=descriptor.supported_platforms,
        visible_platforms=visible_platforms,
    )


def _failure(
    config: RuntimeConfig,
    descriptors: tuple[RuntimeBackendDescriptor, ...],
    *,
    blockers: tuple[RuntimeIssue, ...],
) -> RuntimeSelectionResult:
    missing = tuple(
        item.details["capability"]
        for item in blockers
        if item.code == "runtime_capability_missing" and "capability" in item.details
    )
    return RuntimeSelectionResult(
        status="fail",
        requested_config=config,
        selected_backend=None,
        selected_platform=None,
        considered_backends=descriptors,
        required_capabilities=config.required_capabilities,
        satisfied_capabilities=tuple(
            item for item in config.required_capabilities if item not in missing
        ),
        missing_capabilities=missing,
        fallback_used=False,
        blockers=blockers,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(value)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    result = _sequence(value, name)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return tuple(result)


def _unique_strings(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    result = _strings(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _issues(value: tuple[RuntimeIssue, ...], name: str) -> tuple[RuntimeIssue, ...]:
    result = tuple(value)
    if any(not isinstance(item, RuntimeIssue) for item in result):
        raise TypeError(f"{name} must contain RuntimeIssue values")
    return result


def _issues_from_payload(value: Any, name: str) -> tuple[RuntimeIssue, ...]:
    return tuple(
        RuntimeIssue.from_dict(_mapping(item, name)) for item in _sequence(value, name)
    )
