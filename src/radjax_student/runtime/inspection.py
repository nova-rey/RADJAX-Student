from __future__ import annotations

import importlib
import importlib.util
import platform as host_platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, TypeAlias

from radjax_student.runtime._device_inspection import (
    check_consistency,
    normalize_devices,
)
from radjax_student.runtime.errors import RuntimeIssue
from radjax_student.runtime.models import (
    DeviceInventory,
    RuntimeEnvironment,
)

InspectionStatus: TypeAlias = Literal["pass", "fail"]

RUNTIME_INSPECTION_FINDING_CODES: tuple[str, ...] = (
    "device_memory_unknown",
    "device_normalization_failed",
    "device_precision_unknown",
    "distributed_state_unknown",
    "heterogeneous_platforms_detected",
    "jax_devices_unavailable",
    "jax_global_device_count_unavailable",
    "jax_import_failed",
    "jax_local_device_count_unavailable",
    "jax_not_installed",
    "jax_platform_unavailable",
    "jax_process_count_unavailable",
    "jax_process_index_unavailable",
    "jax_version_unavailable",
    "jaxlib_import_failed",
    "jaxlib_version_unavailable",
    "runtime_inspection_internal_error",
)

RUNTIME_INSPECTION_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "backend_not_registered_or_selected",
    "requested_platform_not_validated",
    "fallback_not_applied",
    "array_creation_not_tested",
    "array_placement_not_tested",
    "compilation_not_tested",
    "synchronization_not_tested",
    "accelerator_execution_not_tested",
    "distributed_execution_not_tested",
    "precision_behavior_not_proven",
    "memory_capacity_not_measured",
    "rng_streams_not_finalized",
    "runtime_state_persistence_not_tested",
    "model_not_allocated",
    "training_not_run",
)


@dataclass(frozen=True)
class RuntimeInspection:
    status: InspectionStatus
    environment: RuntimeEnvironment
    device_inventory: DeviceInventory
    warnings: tuple[RuntimeIssue, ...] = ()
    blockers: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = RUNTIME_INSPECTION_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("runtime inspection status must be 'pass' or 'fail'")
        if not isinstance(self.environment, RuntimeEnvironment):
            raise TypeError("environment must be RuntimeEnvironment")
        if not isinstance(self.device_inventory, DeviceInventory):
            raise TypeError("device_inventory must be DeviceInventory")
        warnings = _issue_tuple(self.warnings, "warnings")
        blockers = _issue_tuple(self.blockers, "blockers")
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        for finding in (*warnings, *blockers):
            if finding.code not in RUNTIME_INSPECTION_FINDING_CODES:
                raise ValueError(f"unknown runtime inspection finding: {finding.code}")
        if self.status == "pass" and blockers:
            raise ValueError("passing runtime inspection cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing runtime inspection must contain blockers")
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "claims_not_made", claims)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "environment": self.environment.to_dict(),
            "device_inventory": self.device_inventory.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeInspection:
        return cls(
            status=str(payload["status"]),
            environment=RuntimeEnvironment.from_dict(
                _mapping(payload["environment"], "environment")
            ),
            device_inventory=DeviceInventory.from_dict(
                _mapping(payload["device_inventory"], "device_inventory")
            ),
            warnings=_issues(payload.get("warnings", ()), "warnings"),
            blockers=_issues(payload.get("blockers", ()), "blockers"),
            claims_not_made=_strings(
                payload.get("claims_not_made", ()),
                "claims_not_made",
            ),
        )


def inspect_runtime_environment() -> RuntimeInspection:
    """Observe local runtime facts without selecting or executing a backend."""

    try:
        return _inspect_runtime_environment()
    except Exception as exc:
        blocker = RuntimeIssue.create(
            "runtime_inspection_internal_error",
            "runtime inspection failed before coherent facts could be produced",
            **_exception_details(exc),
        )
        environment = RuntimeEnvironment(
            python_version=host_platform.python_version(),
            jax_available=False,
            warnings=(blocker.code,),
        )
        return RuntimeInspection(
            status="fail",
            environment=environment,
            device_inventory=DeviceInventory(),
            blockers=(blocker,),
        )


def _inspect_runtime_environment() -> RuntimeInspection:
    warnings: list[RuntimeIssue] = []
    blockers: list[RuntimeIssue] = []
    python_version = host_platform.python_version()
    jax_available = _module_available("jax", warnings)
    if jax_available is not True:
        if jax_available is False:
            warning = RuntimeIssue.create(
                "jax_not_installed",
                "JAX is not installed; runtime observation remains metadata-only",
            )
            warnings.append(warning)
        return _inspection_result(
            python_version=python_version,
            jax_available=False,
            warnings=warnings,
            blockers=blockers,
        )

    try:
        jax_module = importlib.import_module("jax")
    except Exception as exc:
        warnings.append(
            RuntimeIssue.create(
                "jax_import_failed",
                "JAX is installed but could not be imported",
                **_exception_details(exc),
            )
        )
        return _inspection_result(
            python_version=python_version,
            jax_available=False,
            warnings=warnings,
            blockers=blockers,
        )

    jax_version = _module_version(jax_module, "jax_version_unavailable", warnings)
    jaxlib_version = _inspect_jaxlib_version(warnings)
    observed_platform = _safe_jax_string_call(
        jax_module,
        "default_backend",
        "jax_platform_unavailable",
        warnings,
    )
    process_count = _safe_jax_count_call(
        jax_module,
        "process_count",
        "jax_process_count_unavailable",
        warnings,
        blockers,
    )
    process_index = _safe_jax_count_call(
        jax_module,
        "process_index",
        "jax_process_index_unavailable",
        warnings,
        blockers,
    )
    local_device_count = _safe_jax_count_call(
        jax_module,
        "local_device_count",
        "jax_local_device_count_unavailable",
        warnings,
        blockers,
    )
    global_device_count = _safe_jax_count_call(
        jax_module,
        "device_count",
        "jax_global_device_count_unavailable",
        warnings,
        blockers,
    )
    distributed_initialized = _inspect_distributed_state(jax_module, warnings)
    raw_devices = _safe_devices(jax_module, warnings)
    devices = normalize_devices(raw_devices, process_count, warnings, blockers)
    check_consistency(
        devices=devices,
        process_count=process_count,
        process_index=process_index,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        blockers=blockers,
    )
    platforms = tuple(
        sorted({device.platform for device in devices if device.platform is not None})
    )
    if len(platforms) > 1:
        warnings.append(
            RuntimeIssue.create(
                "heterogeneous_platforms_detected",
                "visible devices report more than one platform",
                platforms=platforms,
            )
        )
    inventory = DeviceInventory(
        devices=devices,
        process_count=process_count,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        topology_summary={
            "platforms": platforms,
            "device_kinds": tuple(
                sorted(
                    {
                        device.device_kind
                        for device in devices
                        if device.device_kind is not None
                    }
                )
            ),
            "observation_source": "jax_public_api",
        },
    )
    environment = RuntimeEnvironment(
        python_version=python_version,
        jax_available=True,
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
        platform=observed_platform,
        process_count=process_count,
        process_index=process_index,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        distributed_initialized=distributed_initialized,
        warnings=_finding_codes(warnings),
    )
    return RuntimeInspection(
        status="fail" if blockers else "pass",
        environment=environment,
        device_inventory=inventory,
        warnings=_deduplicate_issues(warnings),
        blockers=_deduplicate_issues(blockers),
    )


def _inspection_result(
    *,
    python_version: str,
    jax_available: bool,
    warnings: list[RuntimeIssue],
    blockers: list[RuntimeIssue],
) -> RuntimeInspection:
    environment = RuntimeEnvironment(
        python_version=python_version,
        jax_available=jax_available,
        warnings=_finding_codes(warnings),
    )
    return RuntimeInspection(
        status="fail" if blockers else "pass",
        environment=environment,
        device_inventory=DeviceInventory(),
        warnings=_deduplicate_issues(warnings),
        blockers=_deduplicate_issues(blockers),
    )


def _module_available(name: str, warnings: list[RuntimeIssue]) -> bool | None:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError) as exc:
        warnings.append(
            RuntimeIssue.create(
                "jax_import_failed",
                "JAX installation could not be discovered reliably",
                **_exception_details(exc),
            )
        )
        return None


def _inspect_jaxlib_version(warnings: list[RuntimeIssue]) -> str | None:
    try:
        available = importlib.util.find_spec("jaxlib") is not None
    except (ImportError, AttributeError, ValueError) as exc:
        warnings.append(
            RuntimeIssue.create(
                "jaxlib_import_failed",
                "JAXLIB installation could not be discovered reliably",
                **_exception_details(exc),
            )
        )
        return None
    if not available:
        warnings.append(
            RuntimeIssue.create(
                "jaxlib_import_failed",
                "JAX imported but JAXLIB is not installed",
                exception_type="ModuleNotFoundError",
            )
        )
        return None
    try:
        module = importlib.import_module("jaxlib")
    except Exception as exc:
        warnings.append(
            RuntimeIssue.create(
                "jaxlib_import_failed",
                "JAXLIB is installed but could not be imported",
                **_exception_details(exc),
            )
        )
        return None
    return _module_version(module, "jaxlib_version_unavailable", warnings)


def _module_version(
    module: ModuleType | Any,
    warning_code: str,
    warnings: list[RuntimeIssue],
) -> str | None:
    value = getattr(module, "__version__", None)
    if isinstance(value, str) and value:
        return value
    warnings.append(
        RuntimeIssue.create(
            warning_code,
            "installed module does not expose a usable public version",
            module=getattr(module, "__name__", "unknown"),
        )
    )
    return None


def _safe_jax_string_call(
    module: Any,
    name: str,
    warning_code: str,
    warnings: list[RuntimeIssue],
) -> str | None:
    value = _safe_call(module, name, warning_code, warnings)
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    warnings.append(
        RuntimeIssue.create(
            warning_code,
            f"JAX {name} returned a non-string value",
            observed_type=type(value).__name__,
        )
    )
    return None


def _safe_jax_count_call(
    module: Any,
    name: str,
    warning_code: str,
    warnings: list[RuntimeIssue],
    blockers: list[RuntimeIssue],
) -> int | None:
    value = _safe_call(module, name, warning_code, warnings)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        blockers.append(
            RuntimeIssue.create(
                "device_normalization_failed",
                f"JAX {name} returned an invalid count",
                field=name,
                observed_type=type(value).__name__,
                observed_value=value if isinstance(value, (str, int)) else None,
            )
        )
        return None
    return value


def _safe_call(
    module: Any,
    name: str,
    warning_code: str,
    warnings: list[RuntimeIssue],
) -> Any:
    function = getattr(module, name, None)
    if not isinstance(function, Callable):
        warnings.append(
            RuntimeIssue.create(
                warning_code,
                f"JAX does not expose callable {name}",
            )
        )
        return None
    try:
        return function()
    except Exception as exc:
        warnings.append(
            RuntimeIssue.create(
                warning_code,
                f"JAX {name} could not be observed",
                **_exception_details(exc),
            )
        )
        return None


def _inspect_distributed_state(
    jax_module: Any,
    warnings: list[RuntimeIssue],
) -> bool | None:
    distributed = getattr(jax_module, "distributed", None)
    function = getattr(distributed, "is_initialized", None)
    if not isinstance(function, Callable):
        warnings.append(
            RuntimeIssue.create(
                "distributed_state_unknown",
                "JAX does not expose distributed initialization state",
            )
        )
        return None
    try:
        value = function()
    except Exception as exc:
        warnings.append(
            RuntimeIssue.create(
                "distributed_state_unknown",
                "distributed initialization state could not be observed",
                **_exception_details(exc),
            )
        )
        return None
    if not isinstance(value, bool):
        warnings.append(
            RuntimeIssue.create(
                "distributed_state_unknown",
                "distributed initialization state was not boolean",
                observed_type=type(value).__name__,
            )
        )
        return None
    return value


def _safe_devices(
    jax_module: Any,
    warnings: list[RuntimeIssue],
) -> tuple[Any, ...]:
    value = _safe_call(
        jax_module,
        "devices",
        "jax_devices_unavailable",
        warnings,
    )
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        warnings.append(
            RuntimeIssue.create(
                "jax_devices_unavailable",
                "JAX devices did not return a sequence",
                observed_type=type(value).__name__,
            )
        )
        return ()
    return tuple(value)


def _exception_details(exc: Exception) -> dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "exception_message": _stable_exception_message(str(exc)),
    }


def _stable_exception_message(message: str) -> str:
    normalized = re.sub(
        r"0x[0-9a-fA-F]+",
        "<memory-address>",
        message,
    )
    normalized = re.sub(
        r"(?<![\w.])/(?:[^\s:'\"]+)",
        "<path>",
        normalized,
    )
    return re.sub(
        r"\b[A-Za-z]:\\[^\s:'\"]+",
        "<path>",
        normalized,
    )


def _finding_codes(findings: list[RuntimeIssue]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(finding.code for finding in findings))


def _deduplicate_issues(findings: list[RuntimeIssue]) -> tuple[RuntimeIssue, ...]:
    result: list[RuntimeIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        key = (finding.code, finding.message, repr(finding.details))
        if key not in seen:
            result.append(finding)
            seen.add(key)
    return tuple(result)


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
