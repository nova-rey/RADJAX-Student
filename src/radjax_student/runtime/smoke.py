"""The small, eager, single-device CPU execution proof introduced in P2.4."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol

from radjax_student.runtime.errors import (
    RUNTIME_ERROR_CODES,
    RuntimeContractError,
    RuntimeIssue,
)
from radjax_student.runtime.inspection import (
    RuntimeInspection,
    inspect_runtime_environment,
)
from radjax_student.runtime.models import (
    DeviceDescriptor,
    ExecutionContext,
    RuntimeConfig,
    freeze_json_mapping,
    json_value,
)
from radjax_student.runtime.registry import (
    RuntimeBackendRegistry,
    build_default_runtime_registry,
)
from radjax_student.runtime.reports import RuntimeReport
from radjax_student.runtime.selection import (
    RuntimeSelectionResult,
    select_runtime_backend,
)

CpuRuntimeSmokeStatus = Literal["pass", "fail"]
CPU_RUNTIME_SMOKE_WARNING_CODES: tuple[str, ...] = (
    "runtime_smoke_not_benchmark",
    "runtime_multiple_cpu_devices_first_selected",
    "runtime_capability_declared_not_proven_beyond_smoke",
)
CPU_RUNTIME_SMOKE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "jit_not_tested",
    "gpu_not_tested",
    "tpu_not_tested",
    "distributed_execution_not_tested",
    "sharding_not_tested",
    "replicated_placement_not_tested",
    "precision_behavior_not_proven",
    "rng_streams_not_consumed_by_smoke",
    "runtime_state_persistence_not_tested",
    "model_parameters_not_initialized",
    "training_not_run",
)


class CpuSmokeBackend(Protocol):
    backend_id: str

    def initialize_cpu_context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection: RuntimeSelectionResult,
        device_descriptor: DeviceDescriptor,
    ) -> ExecutionContext: ...

    def place_cpu_value(self, context: ExecutionContext, value: Any) -> Any: ...

    def execute_cpu_smoke(self, context: ExecutionContext, value: Any) -> Any: ...

    def synchronize_cpu_value(self, context: ExecutionContext, value: Any) -> Any: ...

    def close_cpu_context(self, context: ExecutionContext) -> None: ...


@dataclass(frozen=True)
class CpuRuntimeSmokeReceipt:
    """Serializable receipt for the one allowed P2.4 execution path."""

    status: CpuRuntimeSmokeStatus
    runtime_id: str
    backend_id: str | None
    platform: str | None
    device_id: str | None
    config: RuntimeConfig
    inspection_summary: Mapping[str, Any]
    selection_summary: Mapping[str, Any]
    input_metadata: Mapping[str, Any]
    output_metadata: Mapping[str, Any]
    result_validated: bool
    synchronized: bool
    initialization_seconds: float
    placement_seconds: float
    execution_seconds: float
    synchronization_seconds: float
    teardown_seconds: float
    runtime_report: RuntimeReport
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = CPU_RUNTIME_SMOKE_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail"):
            raise ValueError("CPU smoke status must be pass or fail")
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise ValueError("runtime_id must be a nonempty string")
        for name in ("backend_id", "platform", "device_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string when set")
        if not isinstance(self.config, RuntimeConfig):
            raise TypeError("config must be RuntimeConfig")
        if not isinstance(self.runtime_report, RuntimeReport):
            raise TypeError("runtime_report must be RuntimeReport")
        if not isinstance(self.result_validated, bool) or not isinstance(
            self.synchronized, bool
        ):
            raise TypeError("result_validated and synchronized must be boolean")
        for name in (
            "initialization_seconds",
            "placement_seconds",
            "execution_seconds",
            "synchronization_seconds",
            "teardown_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        blockers = _issues(self.blockers, "blockers")
        warnings = _issues(self.warnings, "warnings")
        invalid_blockers = [
            item.code for item in blockers if item.code not in RUNTIME_ERROR_CODES
        ]
        if invalid_blockers:
            raise ValueError(
                "unknown smoke blocker codes: " + ", ".join(invalid_blockers)
            )
        invalid_warnings = [
            item.code
            for item in warnings
            if item.code not in CPU_RUNTIME_SMOKE_WARNING_CODES
        ]
        if invalid_warnings:
            raise ValueError(
                "unknown smoke warning codes: " + ", ".join(invalid_warnings)
            )
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        if self.status == "pass" and blockers:
            raise ValueError("passing CPU smoke cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing CPU smoke must contain a blocker")
        object.__setattr__(
            self, "inspection_summary", freeze_json_mapping(self.inspection_summary)
        )
        object.__setattr__(
            self, "selection_summary", freeze_json_mapping(self.selection_summary)
        )
        object.__setattr__(
            self, "input_metadata", freeze_json_mapping(self.input_metadata)
        )
        object.__setattr__(
            self, "output_metadata", freeze_json_mapping(self.output_metadata)
        )
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_not_made", claims)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "runtime_id": self.runtime_id,
            "backend_id": self.backend_id,
            "platform": self.platform,
            "device_id": self.device_id,
            "config": self.config.to_dict(),
            "inspection_summary": json_value(self.inspection_summary),
            "selection_summary": json_value(self.selection_summary),
            "input_metadata": json_value(self.input_metadata),
            "output_metadata": json_value(self.output_metadata),
            "result_validated": self.result_validated,
            "synchronized": self.synchronized,
            "initialization_seconds": self.initialization_seconds,
            "placement_seconds": self.placement_seconds,
            "execution_seconds": self.execution_seconds,
            "synchronization_seconds": self.synchronization_seconds,
            "teardown_seconds": self.teardown_seconds,
            "runtime_report": self.runtime_report.to_dict(),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CpuRuntimeSmokeReceipt:
        return cls(
            status=str(payload["status"]),
            runtime_id=str(payload["runtime_id"]),
            backend_id=_optional_string(payload.get("backend_id")),
            platform=_optional_string(payload.get("platform")),
            device_id=_optional_string(payload.get("device_id")),
            config=RuntimeConfig.from_dict(_mapping(payload["config"], "config")),
            inspection_summary=_mapping(
                payload["inspection_summary"], "inspection_summary"
            ),
            selection_summary=_mapping(
                payload["selection_summary"], "selection_summary"
            ),
            input_metadata=_mapping(payload["input_metadata"], "input_metadata"),
            output_metadata=_mapping(payload["output_metadata"], "output_metadata"),
            result_validated=_bool(payload["result_validated"], "result_validated"),
            synchronized=_bool(payload["synchronized"], "synchronized"),
            initialization_seconds=_number(
                payload["initialization_seconds"], "initialization_seconds"
            ),
            placement_seconds=_number(
                payload["placement_seconds"], "placement_seconds"
            ),
            execution_seconds=_number(
                payload["execution_seconds"], "execution_seconds"
            ),
            synchronization_seconds=_number(
                payload["synchronization_seconds"], "synchronization_seconds"
            ),
            teardown_seconds=_number(payload["teardown_seconds"], "teardown_seconds"),
            runtime_report=RuntimeReport.from_dict(
                _mapping(payload["runtime_report"], "runtime_report")
            ),
            blockers=_issues_from_payload(payload.get("blockers", ()), "blockers"),
            warnings=_issues_from_payload(payload.get("warnings", ()), "warnings"),
            claims_not_made=_strings(
                payload.get("claims_not_made", ()), "claims_not_made"
            ),
        )


def default_cpu_smoke_config() -> RuntimeConfig:
    return RuntimeConfig(
        backend_id="jax",
        platform_preference="cpu",
        placement_policy="single_device",
        compilation_policy="eager",
        distributed_policy="disabled",
        fallback_policy="disallowed",
        seed=0,
    )


def run_single_device_cpu_smoke(
    config: RuntimeConfig | None = None,
    *,
    inspection: RuntimeInspection | None = None,
    registry: RuntimeBackendRegistry | None = None,
) -> CpuRuntimeSmokeReceipt:
    """Inspect, select, and run the one eager CPU-only runtime heartbeat."""

    selected_config = default_cpu_smoke_config() if config is None else config
    selected_inspection = (
        inspect_runtime_environment() if inspection is None else inspection
    )
    selected_registry = (
        build_default_runtime_registry() if registry is None else registry
    )
    selection = select_runtime_backend(
        selected_config,
        selected_inspection,
        selected_registry,
    )
    return run_selected_cpu_smoke(
        config=selected_config,
        inspection=selected_inspection,
        selection=selection,
        backend=selected_registry.get("jax"),
    )


def run_selected_cpu_smoke(
    *,
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    selection: RuntimeSelectionResult,
    backend: Any,
) -> CpuRuntimeSmokeReceipt:
    """Execute only after the supplied inspection and selection approve JAX CPU."""

    runtime_id = f"jax-cpu-smoke-seed-{config.seed}"
    blockers = _preflight_blockers(config, inspection, selection, backend)
    device = _selected_cpu_descriptor(inspection)
    if device is None:
        blockers.append(
            RuntimeIssue.create(
                "runtime_single_device_required",
                "P2.4 requires one visible CPU device to select",
            )
        )
    warnings = _smoke_warnings(inspection)
    if blockers:
        return _receipt(
            status="fail",
            runtime_id=runtime_id,
            config=config,
            inspection=inspection,
            selection=selection,
            device=device,
            blockers=tuple(blockers),
            warnings=warnings,
        )

    assert device is not None
    context: ExecutionContext | None = None
    placed: Any = None
    result: Any = None
    input_metadata = {"shape": (3,), "dtype": "float32", "values": (1.0, 2.0, 3.0)}
    output_metadata: Mapping[str, Any] = {}
    result_validated = False
    synchronized = False
    timings = _Timings()
    try:
        start = time.perf_counter()
        try:
            context = backend.initialize_cpu_context(
                config, inspection, selection, device
            )
        except Exception as exc:
            blockers.append(
                _issue_from_exception(
                    "runtime_backend_initialization_failed",
                    "runtime backend initialization failed",
                    exc,
                )
            )
        timings.initialization_seconds = time.perf_counter() - start

        if not blockers:
            start = time.perf_counter()
            try:
                placed = backend.place_cpu_value(context, input_metadata["values"])
                input_metadata = _array_metadata(placed, input_metadata["values"])
            except Exception as exc:
                blockers.append(
                    _issue_from_exception(
                        "runtime_placement_failed", "CPU value placement failed", exc
                    )
                )
            timings.placement_seconds = time.perf_counter() - start

        if not blockers:
            start = time.perf_counter()
            try:
                result = backend.execute_cpu_smoke(context, placed)
            except Exception as exc:
                blockers.append(
                    _issue_from_exception(
                        "runtime_execution_failed", "CPU smoke execution failed", exc
                    )
                )
            timings.execution_seconds = time.perf_counter() - start

        if not blockers:
            start = time.perf_counter()
            try:
                result = backend.synchronize_cpu_value(context, result)
                synchronized = True
            except Exception as exc:
                blockers.append(
                    _issue_from_exception(
                        "runtime_synchronization_failed",
                        "CPU result synchronization failed",
                        exc,
                    )
                )
            timings.synchronization_seconds = time.perf_counter() - start

        if not blockers:
            output_metadata = _array_metadata(result, ())
            observed = _host_values(result)
            expected = (3.0, 5.0, 7.0)
            if observed != expected:
                blockers.append(
                    RuntimeIssue.create(
                        "runtime_smoke_result_mismatch",
                        "CPU smoke output did not match the deterministic "
                        "expected value",
                        expected=expected,
                        observed=observed,
                    )
                )
            else:
                result_validated = True
    except Exception as exc:
        blockers.append(
            _issue_from_exception(
                "runtime_smoke_internal_error",
                "CPU smoke failed outside a recognized execution phase",
                exc,
            )
        )
    finally:
        if context is not None:
            start = time.perf_counter()
            try:
                backend.close_cpu_context(context)
            except Exception as exc:
                blockers.append(
                    _issue_from_exception(
                        "runtime_teardown_failed",
                        "CPU smoke teardown failed",
                        exc,
                    )
                )
            timings.teardown_seconds = time.perf_counter() - start

    return _receipt(
        status="pass" if not blockers else "fail",
        runtime_id=runtime_id,
        config=config,
        inspection=inspection,
        selection=selection,
        device=device,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
        result_validated=result_validated,
        synchronized=synchronized,
        timings=timings,
        blockers=tuple(blockers),
        warnings=warnings,
    )


@dataclass
class _Timings:
    initialization_seconds: float = 0.0
    placement_seconds: float = 0.0
    execution_seconds: float = 0.0
    synchronization_seconds: float = 0.0
    teardown_seconds: float = 0.0


def _preflight_blockers(
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    selection: RuntimeSelectionResult,
    backend: Any,
) -> list[RuntimeIssue]:
    blockers: list[RuntimeIssue] = []
    if config.backend_id != "jax" or config.platform_preference != "cpu":
        blockers.append(
            RuntimeIssue.create(
                "runtime_cpu_platform_required",
                "P2.4 requires an explicit JAX CPU configuration",
                backend_id=config.backend_id,
                platform_preference=config.platform_preference,
            )
        )
    if (
        config.placement_policy != "single_device"
        or config.compilation_policy != "eager"
        or config.distributed_policy != "disabled"
        or config.fallback_policy != "disallowed"
    ):
        blockers.append(
            RuntimeIssue.create(
                "runtime_backend_ineligible",
                "P2.4 supports only eager, single-device, non-distributed JAX "
                "CPU execution",
            )
        )
    if inspection.environment.process_count != 1:
        blockers.append(
            RuntimeIssue.create(
                "runtime_single_process_required",
                "P2.4 requires exactly one observed process",
                process_count=inspection.environment.process_count,
            )
        )
    if not selection.ok:
        blockers.extend(selection.blockers)
    elif (
        selection.selected_backend is None
        or selection.selected_backend.backend_id != "jax"
        or selection.selected_platform != "cpu"
    ):
        blockers.append(
            RuntimeIssue.create(
                "runtime_backend_ineligible",
                "P2.4 requires a selected JAX CPU backend",
            )
        )
    if backend is None or getattr(backend, "backend_id", None) != "jax":
        blockers.append(
            RuntimeIssue.create(
                "runtime_backend_ineligible",
                "selected JAX backend implementation is unavailable",
            )
        )
    return _deduplicate_issues(blockers)


def _selected_cpu_descriptor(inspection: RuntimeInspection) -> DeviceDescriptor | None:
    candidates = sorted(
        (
            item
            for item in inspection.device_inventory.devices
            if item.platform == "cpu"
        ),
        key=lambda item: item.device_id,
    )
    return candidates[0] if candidates else None


def _smoke_warnings(inspection: RuntimeInspection) -> tuple[RuntimeIssue, ...]:
    cpu_devices = tuple(
        item for item in inspection.device_inventory.devices if item.platform == "cpu"
    )
    warnings = [
        RuntimeIssue.create(
            "runtime_smoke_not_benchmark",
            "P2.4 timings are diagnostic execution observations, not benchmarks",
        ),
        RuntimeIssue.create(
            "runtime_capability_declared_not_proven_beyond_smoke",
            "the CPU smoke does not prove capabilities beyond eager "
            "single-device execution",
        ),
    ]
    if len(cpu_devices) > 1:
        warnings.append(
            RuntimeIssue.create(
                "runtime_multiple_cpu_devices_first_selected",
                "multiple CPU devices were visible; the stable first device was "
                "selected",
                device_ids=tuple(sorted(item.device_id for item in cpu_devices)),
            )
        )
    return tuple(warnings)


def _receipt(
    *,
    status: CpuRuntimeSmokeStatus,
    runtime_id: str,
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    selection: RuntimeSelectionResult,
    device: DeviceDescriptor | None,
    input_metadata: Mapping[str, Any] = MappingProxyType({}),
    output_metadata: Mapping[str, Any] = MappingProxyType({}),
    result_validated: bool = False,
    synchronized: bool = False,
    timings: _Timings | None = None,
    blockers: tuple[RuntimeIssue, ...],
    warnings: tuple[RuntimeIssue, ...],
) -> CpuRuntimeSmokeReceipt:
    current_timings = _Timings() if timings is None else timings
    backend_id = (
        None
        if selection.selected_backend is None
        else selection.selected_backend.backend_id
    )
    capabilities = (
        None
        if selection.selected_backend is None
        else selection.selected_backend.capability_profile
    )
    runtime_report = RuntimeReport(
        status=status,
        backend_id=backend_id,
        environment=inspection.environment,
        device_inventory=inspection.device_inventory,
        capabilities=capabilities,
        selected_policy=config,
        blockers=blockers,
        warnings=warnings,
        claims_not_made=CPU_RUNTIME_SMOKE_CLAIMS_NOT_MADE,
    )
    return CpuRuntimeSmokeReceipt(
        status=status,
        runtime_id=runtime_id,
        backend_id=backend_id,
        platform="cpu" if device is not None else None,
        device_id=None if device is None else device.device_id,
        config=config,
        inspection_summary={
            "status": inspection.status,
            "jax_available": inspection.environment.jax_available,
            "process_count": inspection.environment.process_count,
            "visible_platforms": tuple(
                sorted(
                    {
                        item.platform
                        for item in inspection.device_inventory.devices
                        if item.platform is not None
                    }
                )
            ),
        },
        selection_summary={
            "status": selection.status,
            "selected_backend_id": backend_id,
            "selected_platform": selection.selected_platform,
            "fallback_used": selection.fallback_used,
            "blocker_codes": tuple(item.code for item in selection.blockers),
            "warning_codes": tuple(item.code for item in selection.warnings),
        },
        input_metadata=input_metadata,
        output_metadata=output_metadata,
        result_validated=result_validated,
        synchronized=synchronized,
        initialization_seconds=current_timings.initialization_seconds,
        placement_seconds=current_timings.placement_seconds,
        execution_seconds=current_timings.execution_seconds,
        synchronization_seconds=current_timings.synchronization_seconds,
        teardown_seconds=current_timings.teardown_seconds,
        runtime_report=runtime_report,
        blockers=blockers,
        warnings=warnings,
    )


def _array_metadata(value: Any, fallback_values: tuple[float, ...]) -> dict[str, Any]:
    shape = getattr(value, "shape", (len(fallback_values),))
    dtype = getattr(value, "dtype", "unknown")
    return {
        "shape": tuple(int(item) for item in shape),
        "dtype": str(dtype),
    }


def _host_values(value: Any) -> tuple[float, ...]:
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)):
        raise TypeError("smoke result is not a host sequence")
    return tuple(float(item) for item in raw)


def _issue_from_exception(code: str, message: str, exc: Exception) -> RuntimeIssue:
    if isinstance(exc, RuntimeContractError):
        return exc.issue
    return RuntimeIssue.create(
        code,
        message,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
    )


def _deduplicate_issues(issues: list[RuntimeIssue]) -> list[RuntimeIssue]:
    result: list[RuntimeIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        identity = (issue.code, issue.message, repr(sorted(issue.details.items())))
        if identity not in seen:
            result.append(issue)
            seen.add(identity)
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional value must be a string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


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


def _issues(value: Any, name: str) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, RuntimeIssue) for item in result):
        raise TypeError(f"{name} must contain RuntimeIssue values")
    return result


def _issues_from_payload(value: Any, name: str) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(RuntimeIssue.from_dict(_mapping(item, name)) for item in value)
