"""One shared CPU, GPU, and TPU portability smoke path for the runtime."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.runtime.errors import RuntimeIssue
from radjax_student.runtime.execution import (
    ExecutionRequest,
    execute_function,
)
from radjax_student.runtime.inspection import (
    RuntimeInspection,
    inspect_runtime_environment,
)
from radjax_student.runtime.models import (
    CompilationOptions,
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
from radjax_student.runtime.selection import select_runtime_backend
from radjax_student.runtime.state import (
    evaluate_runtime_resume_compatibility,
    load_runtime_state_with_receipt,
    runtime_state_from_context,
    save_runtime_state,
)

PortabilityPlatform = Literal["cpu", "gpu", "tpu"]
PortabilityMode = Literal["eager", "jit"]
PortabilityStatus = Literal["pass", "unavailable", "fail"]
PORTABILITY_PLATFORMS: tuple[PortabilityPlatform, ...] = ("cpu", "gpu", "tpu")
PORTABILITY_MODES: tuple[PortabilityMode, ...] = ("eager", "jit")
PORTABILITY_BLOCKER_CODES: tuple[str, ...] = (
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
)
PORTABILITY_WARNING_CODES: tuple[str, ...] = (
    "runtime_portability_single_device_only",
    "runtime_portability_not_benchmark",
    "runtime_portability_accelerator_receipt_external",
    "runtime_portability_topology_not_migrated",
    "runtime_portability_precision_not_exhaustively_tested",
    "runtime_portability_jit_optional",
)
PORTABILITY_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "multi_device_execution_not_tested",
    "distributed_execution_not_tested",
    "sharding_not_tested",
    "replicated_placement_not_tested",
    "training_not_run",
    "gradients_not_computed",
    "optimizer_not_updated",
    "performance_not_benchmarked",
    "cross_target_numerical_identity_not_proven",
    "model_quality_not_claimed",
)


@dataclass(frozen=True)
class RuntimePortabilityTimings:
    initialization_seconds: float = 0.0
    placement_seconds: float = 0.0
    execution_seconds: float = 0.0
    synchronization_seconds: float = 0.0
    state_seconds: float = 0.0
    teardown_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "initialization_seconds",
            "placement_seconds",
            "execution_seconds",
            "synchronization_seconds",
            "state_seconds",
            "teardown_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be numeric")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, float]:
        return {
            "initialization_seconds": float(self.initialization_seconds),
            "placement_seconds": float(self.placement_seconds),
            "execution_seconds": float(self.execution_seconds),
            "synchronization_seconds": float(self.synchronization_seconds),
            "state_seconds": float(self.state_seconds),
            "teardown_seconds": float(self.teardown_seconds),
        }


@dataclass(frozen=True)
class RuntimePortabilityReceipt:
    status: PortabilityStatus
    platform: PortabilityPlatform
    backend_id: str | None
    device_id: str | None
    process_count: int | None
    local_device_count: int | None
    global_device_count: int | None
    execution_mode: PortabilityMode
    placement_policy: str
    result_validated: bool
    synchronized: bool
    runtime_state_round_trip: bool
    timings: RuntimePortabilityTimings = RuntimePortabilityTimings()
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = PORTABILITY_CLAIMS_NOT_MADE
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.status not in ("pass", "unavailable", "fail"):
            raise ValueError("portability status must be pass, unavailable, or fail")
        if self.platform not in PORTABILITY_PLATFORMS:
            raise ValueError("portability platform must be cpu, gpu, or tpu")
        if self.execution_mode not in PORTABILITY_MODES:
            raise ValueError("portability execution mode must be eager or jit")
        for name in ("backend_id", "device_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string when present")
        for name in (
            "process_count",
            "local_device_count",
            "global_device_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer when present")
        if not isinstance(self.placement_policy, str) or not self.placement_policy:
            raise ValueError("placement_policy must be a nonempty string")
        for name in ("result_validated", "synchronized", "runtime_state_round_trip"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.timings, RuntimePortabilityTimings):
            raise TypeError("timings must be RuntimePortabilityTimings")
        blockers = _issues(self.blockers, "blockers")
        warnings = _issues(self.warnings, "warnings")
        if any(item.code not in PORTABILITY_BLOCKER_CODES for item in blockers):
            raise ValueError("unknown portability blocker code")
        if any(item.code not in PORTABILITY_WARNING_CODES for item in warnings):
            raise ValueError("unknown portability warning code")
        if self.status == "pass" and blockers:
            raise ValueError("passing portability smoke cannot contain blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing portability smoke must contain blockers")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "claims_not_made",
            _unique_strings(self.claims_not_made, "claims_not_made"),
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def unavailable(self) -> bool:
        return self.status == "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "platform": self.platform,
            "backend_id": self.backend_id,
            "device_id": self.device_id,
            "process_count": self.process_count,
            "local_device_count": self.local_device_count,
            "global_device_count": self.global_device_count,
            "execution_mode": self.execution_mode,
            "placement_policy": self.placement_policy,
            "result_validated": self.result_validated,
            "synchronized": self.synchronized,
            "runtime_state_round_trip": self.runtime_state_round_trip,
            "timings": self.timings.to_dict(),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
            "metadata": json_value(self.metadata),
        }


def default_portability_config(
    platform: PortabilityPlatform,
    mode: PortabilityMode = "eager",
) -> RuntimeConfig:
    """Return explicit one-device JAX intent for the selected portability target."""

    _validate_platform(platform)
    _validate_mode(mode)
    return RuntimeConfig(
        backend_id="jax",
        platform_preference=platform,
        placement_policy="single_device",
        compilation_policy=mode,
        distributed_policy="disabled",
        fallback_policy="disallowed",
        seed=0,
    )


def run_portability_smoke(
    platform: PortabilityPlatform,
    mode: PortabilityMode = "eager",
    *,
    config: RuntimeConfig | None = None,
    inspection: RuntimeInspection | None = None,
    registry: RuntimeBackendRegistry | None = None,
) -> RuntimePortabilityReceipt:
    """Execute one shared architecture-independent smoke on one requested target."""

    _validate_platform(platform)
    _validate_mode(mode)
    selected_config = (
        default_portability_config(platform, mode) if config is None else config
    )
    if selected_config.platform_preference != platform:
        raise ValueError("portability config platform_preference must match platform")
    if selected_config.compilation_policy != mode:
        raise ValueError("portability config compilation_policy must match mode")
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
    if not selection.ok or selection.selected_backend is None:
        return _unavailable_receipt(
            platform=platform,
            mode=mode,
            config=selected_config,
            inspection=selected_inspection,
            selection_blockers=selection.blockers,
        )
    backend = selected_registry.get(selection.selected_backend.backend_id)
    if backend is None:
        return _failure_receipt(
            platform=platform,
            mode=mode,
            config=selected_config,
            inspection=selected_inspection,
            backend_id=selection.selected_backend.backend_id,
            blocker=_issue(
                "runtime_portability_backend_unavailable",
                "selected portability backend is no longer registered",
            ),
        )
    device = _selected_device(selected_inspection, platform)
    if device is None:
        return _unavailable_receipt(
            platform=platform,
            mode=mode,
            config=selected_config,
            inspection=selected_inspection,
            backend_id=selection.selected_backend.backend_id,
            selection_blockers=(),
            blocker_code="runtime_portability_device_unavailable",
        )
    return _run_selected_portability_smoke(
        platform=platform,
        mode=mode,
        config=selected_config,
        inspection=selected_inspection,
        selection=selection,
        backend=backend,
        device=device,
    )


def _run_selected_portability_smoke(
    *,
    platform: PortabilityPlatform,
    mode: PortabilityMode,
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    selection: Any,
    backend: Any,
    device: DeviceDescriptor,
) -> RuntimePortabilityReceipt:
    timings = _MutableTimings()
    context: ExecutionContext | None = None
    backend_id = getattr(backend, "backend_id", None)
    blockers: list[RuntimeIssue] = []
    result_validated = False
    synchronized = False
    runtime_state_round_trip = False
    try:
        start = time.perf_counter()
        try:
            context = backend.initialize_portability_context(
                config, inspection, selection, device
            )
        except Exception as exc:
            blockers.append(
                _exception_issue(
                    "runtime_portability_initialization_failed",
                    "selected target context could not be initialized",
                    exc,
                )
            )
        finally:
            timings.initialization_seconds = time.perf_counter() - start

        if not blockers:
            start = time.perf_counter()
            try:
                placed = backend.place_portability_value(context, (1.0, 2.0, 3.0))
            except Exception as exc:
                blockers.append(
                    _exception_issue(
                        "runtime_portability_placement_failed",
                        "selected target value could not be explicitly placed",
                        exc,
                    )
                )
            finally:
                timings.placement_seconds = time.perf_counter() - start

        if not blockers:
            request = ExecutionRequest(
                request_id=f"portability-{platform}-{mode}-v1",
                function_id="runtime.portability_scale_add",
                mode=mode,
                compilation_options=CompilationOptions(
                    mode=mode,
                    synchronize_results=True,
                ),
            )
            output, execution = execute_function(
                context=context,
                function=_scale_add,
                request=request,
                backend=backend,
                args=(placed,),
            )
            timings.execution_seconds = (
                execution.preparation_seconds
                + execution.compilation_seconds
                + execution.dispatch_seconds
            )
            timings.synchronization_seconds = execution.synchronization_seconds
            if not execution.ok:
                blockers.append(
                    _issue(
                        "runtime_portability_execution_failed",
                        "shared P2.7 execution boundary failed on selected target",
                        execution_blockers=tuple(
                            item.code for item in execution.blockers
                        ),
                    )
                )
            elif not execution.synchronized:
                blockers.append(
                    _issue(
                        "runtime_portability_synchronization_failed",
                        "shared P2.7 execution did not confirm target completion",
                    )
                )
            else:
                synchronized = True
                if _expected_output(output):
                    result_validated = True
                else:
                    blockers.append(
                        _issue(
                            "runtime_portability_result_mismatch",
                            "selected target result differs from the shared smoke "
                            "expectation",
                        )
                    )

        if not blockers:
            start = time.perf_counter()
            try:
                state = runtime_state_from_context(
                    context,
                    config,
                    global_step=3,
                    resume_metadata={"portability_platform": platform},
                )
                with tempfile.TemporaryDirectory(
                    prefix="radjax-portability-"
                ) as temp_dir:
                    state_dir = Path(temp_dir) / "runtime_state"
                    save_runtime_state(state, state_dir)
                    loaded, receipt = load_runtime_state_with_receipt(state_dir)
                compatibility = evaluate_runtime_resume_compatibility(
                    loaded,
                    config,
                    inspection,
                )
                runtime_state_round_trip = (
                    loaded == state
                    and bool(receipt.verified_files)
                    and compatibility.ok
                )
                if not runtime_state_round_trip:
                    blockers.append(
                        _issue(
                            "runtime_portability_state_round_trip_failed",
                            "runtime metadata round trip did not preserve compatible "
                            "state",
                        )
                    )
            except Exception as exc:
                blockers.append(
                    _exception_issue(
                        "runtime_portability_state_round_trip_failed",
                        "runtime metadata save/restore failed on selected target",
                        exc,
                    )
                )
            finally:
                timings.state_seconds = time.perf_counter() - start
    except Exception as exc:
        blockers.append(
            _exception_issue(
                "runtime_portability_internal_error",
                "portability smoke failed outside a recognized phase",
                exc,
            )
        )
    finally:
        if context is not None:
            start = time.perf_counter()
            try:
                backend.close_portability_context(context)
            except Exception as exc:
                blockers.append(
                    _exception_issue(
                        "runtime_portability_teardown_failed",
                        "selected target context could not be torn down cleanly",
                        exc,
                    )
                )
            finally:
                timings.teardown_seconds = time.perf_counter() - start
    return RuntimePortabilityReceipt(
        status="fail" if blockers else "pass",
        platform=platform,
        backend_id=backend_id,
        device_id=device.device_id,
        process_count=inspection.environment.process_count,
        local_device_count=inspection.device_inventory.local_device_count,
        global_device_count=inspection.device_inventory.global_device_count,
        execution_mode=mode,
        placement_policy=config.placement_policy,
        result_validated=result_validated,
        synchronized=synchronized,
        runtime_state_round_trip=runtime_state_round_trip,
        timings=timings.freeze(),
        blockers=tuple(blockers),
        warnings=_warnings(platform, mode, inspection),
        metadata=_receipt_metadata(inspection, device),
    )


def _unavailable_receipt(
    *,
    platform: PortabilityPlatform,
    mode: PortabilityMode,
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    selection_blockers: tuple[RuntimeIssue, ...],
    backend_id: str | None = None,
    blocker_code: str | None = None,
) -> RuntimePortabilityReceipt:
    code = blocker_code or _unavailable_code(selection_blockers)
    return RuntimePortabilityReceipt(
        status="unavailable",
        platform=platform,
        backend_id=backend_id,
        device_id=None,
        process_count=inspection.environment.process_count,
        local_device_count=inspection.device_inventory.local_device_count,
        global_device_count=inspection.device_inventory.global_device_count,
        execution_mode=mode,
        placement_policy=config.placement_policy,
        result_validated=False,
        synchronized=False,
        runtime_state_round_trip=False,
        blockers=(
            _issue(
                code,
                "requested portability target is not available in the observed "
                "environment",
                platform=platform,
                selection_blockers=tuple(item.code for item in selection_blockers),
            ),
        ),
        warnings=_warnings(platform, mode, inspection),
        metadata=_receipt_metadata(inspection, None),
    )


def _failure_receipt(
    *,
    platform: PortabilityPlatform,
    mode: PortabilityMode,
    config: RuntimeConfig,
    inspection: RuntimeInspection,
    blocker: RuntimeIssue,
    backend_id: str | None = None,
    device_id: str | None = None,
    timings: RuntimePortabilityTimings | None = None,
) -> RuntimePortabilityReceipt:
    return RuntimePortabilityReceipt(
        status="fail",
        platform=platform,
        backend_id=backend_id,
        device_id=device_id,
        process_count=inspection.environment.process_count,
        local_device_count=inspection.device_inventory.local_device_count,
        global_device_count=inspection.device_inventory.global_device_count,
        execution_mode=mode,
        placement_policy=config.placement_policy,
        result_validated=False,
        synchronized=False,
        runtime_state_round_trip=False,
        timings=RuntimePortabilityTimings() if timings is None else timings,
        blockers=(blocker,),
        warnings=_warnings(platform, mode, inspection),
        metadata=_receipt_metadata(inspection, None),
    )


def _selected_device(
    inspection: RuntimeInspection,
    platform: PortabilityPlatform,
) -> DeviceDescriptor | None:
    candidates = [
        device
        for device in inspection.device_inventory.devices
        if device.platform == platform
        and (
            inspection.environment.process_index is None
            or device.process_index in (None, inspection.environment.process_index)
        )
    ]
    return min(candidates, key=lambda item: item.device_id) if candidates else None


def _scale_add(value: Any) -> Any:
    return value * 2 + 1


def _expected_output(value: Any) -> bool:
    try:
        return tuple(float(item) for item in value) == (3.0, 5.0, 7.0)
    except (TypeError, ValueError):
        return False


def _unavailable_code(blockers: tuple[RuntimeIssue, ...]) -> str:
    if any(item.code == "runtime_backend_unavailable" for item in blockers):
        return "runtime_portability_backend_unavailable"
    return "runtime_portability_platform_unavailable"


def _warnings(
    platform: PortabilityPlatform,
    mode: PortabilityMode,
    inspection: RuntimeInspection,
) -> tuple[RuntimeIssue, ...]:
    warnings = [
        _issue(
            "runtime_portability_single_device_only",
            "P2.9 selects one local device and does not create meshes or sharding",
        ),
        _issue(
            "runtime_portability_not_benchmark",
            "portability timings are diagnostic observations, not benchmarks",
        ),
        _issue(
            "runtime_portability_topology_not_migrated",
            "saved topology remains historical metadata and is not migrated",
        ),
        _issue(
            "runtime_portability_precision_not_exhaustively_tested",
            "the smoke does not exhaustively prove target precision behavior",
        ),
    ]
    if platform in ("gpu", "tpu"):
        warnings.append(
            _issue(
                "runtime_portability_accelerator_receipt_external",
                "accelerator receipt is intended for an environment with this target",
                platform=platform,
            )
        )
    if mode == "eager":
        warnings.append(
            _issue(
                "runtime_portability_jit_optional",
                "JIT portability is optional additional coverage",
            )
        )
    if inspection.environment.process_count not in (None, 1):
        warnings.append(
            _issue(
                "runtime_portability_single_device_only",
                "multi-process observation does not enable distributed portability "
                "execution",
            )
        )
    return tuple(warnings)


def _receipt_metadata(
    inspection: RuntimeInspection,
    device: DeviceDescriptor | None,
) -> Mapping[str, Any]:
    return {
        "jax_version": inspection.environment.jax_version,
        "jaxlib_version": inspection.environment.jaxlib_version,
        "device_kind": None if device is None else device.device_kind,
        "selected_device_process_index": None
        if device is None
        else device.process_index,
    }


def _issue(code: str, message: str, **details: Any) -> RuntimeIssue:
    return RuntimeIssue.create(code, message, **details)


def _exception_issue(code: str, message: str, exc: Exception) -> RuntimeIssue:
    return _issue(code, message, exception_type=type(exc).__name__)


def _validate_platform(platform: str) -> None:
    if platform not in PORTABILITY_PLATFORMS:
        raise ValueError("platform must be one of cpu, gpu, or tpu")


def _validate_mode(mode: str) -> None:
    if mode not in PORTABILITY_MODES:
        raise ValueError("mode must be eager or jit")


def _issues(value: Any, name: str) -> tuple[RuntimeIssue, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, RuntimeIssue) for item in result):
        raise TypeError(f"{name} must contain RuntimeIssue values")
    return result


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass
class _MutableTimings:
    initialization_seconds: float = 0.0
    placement_seconds: float = 0.0
    execution_seconds: float = 0.0
    synchronization_seconds: float = 0.0
    state_seconds: float = 0.0
    teardown_seconds: float = 0.0

    def freeze(self) -> RuntimePortabilityTimings:
        return RuntimePortabilityTimings(
            initialization_seconds=self.initialization_seconds,
            placement_seconds=self.placement_seconds,
            execution_seconds=self.execution_seconds,
            synchronization_seconds=self.synchronization_seconds,
            state_seconds=self.state_seconds,
            teardown_seconds=self.teardown_seconds,
        )
