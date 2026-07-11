from __future__ import annotations

import json
import socket
import urllib.request
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from radjax_student.runtime import (
    DeviceDescriptor,
    DeviceInventory,
    ExecutionContext,
    RuntimeBackendAvailability,
    RuntimeBackendRegistry,
    RuntimeCapabilityProfile,
    RuntimeConfig,
    RuntimeEnvironment,
    RuntimeInspection,
    run_portability_smoke,
)
from radjax_student.runtime import portability as portability_module


@pytest.mark.parametrize("platform", ("cpu", "gpu", "tpu"))
def test_shared_portability_path_passes_on_fake_targets(platform: str) -> None:
    backend = _FakePortabilityBackend(platforms=(platform,))
    inspection = _inspection(platform)

    receipt = run_portability_smoke(
        platform,
        config=_config(platform),
        inspection=inspection,
        registry=_registry(backend),
    )

    assert receipt.status == "pass"
    assert receipt.platform == platform
    assert receipt.device_id == f"{platform}:0"
    assert receipt.result_validated
    assert receipt.synchronized
    assert receipt.runtime_state_round_trip
    assert backend.placed_device_ids == [f"{platform}:0"]
    assert backend.closed_context_ids == [f"fake-{platform}-context"]
    assert receipt.timings.teardown_seconds >= 0.0
    assert receipt.to_dict()["timings"]["teardown_seconds"] == pytest.approx(
        receipt.timings.teardown_seconds
    )
    assert json.dumps(receipt.to_dict(), sort_keys=True) == json.dumps(
        receipt.to_dict(), sort_keys=True
    )


@pytest.mark.parametrize("platform", ("gpu", "tpu"))
def test_missing_accelerator_is_honestly_unavailable(platform: str) -> None:
    backend = _FakePortabilityBackend(platforms=("cpu", "gpu", "tpu"))
    receipt = run_portability_smoke(
        platform,
        config=_config(platform),
        inspection=_inspection("cpu"),
        registry=_registry(backend),
    )

    assert receipt.status == "unavailable"
    assert receipt.ok is False
    assert receipt.result_validated is False
    assert receipt.runtime_state_round_trip is False
    assert receipt.blockers[0].code == "runtime_portability_platform_unavailable"


def test_portability_jit_uses_the_same_execution_path() -> None:
    backend = _FakePortabilityBackend(platforms=("cpu",))
    receipt = run_portability_smoke(
        "cpu",
        mode="jit",
        config=_config("cpu", mode="jit"),
        inspection=_inspection("cpu"),
        registry=_registry(backend),
    )

    assert receipt.status == "pass"
    assert receipt.execution_mode == "jit"
    assert backend.compilation_modes == ["jit"]


def test_doctor_portability_option_embeds_the_explicit_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radjax_student.cli.main import main
    from radjax_student.reports import doctor as doctor_module

    backend = _FakePortabilityBackend(platforms=("gpu",))
    receipt = run_portability_smoke(
        "gpu",
        config=_config("gpu"),
        inspection=_inspection("gpu"),
        registry=_registry(backend),
    )
    monkeypatch.setattr(
        doctor_module,
        "run_portability_smoke",
        lambda platform, mode: receipt,
    )

    stdout = StringIO()
    code = main(
        ("doctor", "--portability-smoke", "gpu", "--format", "json"),
        stdout=stdout,
    )
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["runtime_portability_smoke"]["status"] == "pass"
    assert payload["runtime_portability_smoke"]["platform"] == "gpu"


def test_portability_smoke_does_not_use_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    backend = _FakePortabilityBackend(platforms=("cpu",))

    receipt = run_portability_smoke(
        "cpu",
        config=_config("cpu"),
        inspection=_inspection("cpu"),
        registry=_registry(backend),
    )

    assert receipt.ok


@pytest.mark.parametrize(
    ("failure_phase", "expected_blocker"),
    (
        ("placement", "runtime_portability_placement_failed"),
        ("execution", "runtime_portability_execution_failed"),
        ("synchronization", "runtime_portability_execution_failed"),
    ),
)
def test_teardown_runs_after_runtime_phase_failure(
    failure_phase: str,
    expected_blocker: str,
) -> None:
    backend = _FakePortabilityBackend(
        platforms=("cpu",),
        failure_phase=failure_phase,
    )

    receipt = _run_fake(backend)

    assert receipt.status == "fail"
    assert receipt.blockers[0].code == expected_blocker
    assert backend.closed_context_ids == ["fake-cpu-context"]
    assert receipt.timings.teardown_seconds >= 0.0


def test_teardown_runs_after_runtime_state_round_trip_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakePortabilityBackend(platforms=("cpu",))

    def fail_state(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("controlled state failure")

    monkeypatch.setattr(portability_module, "save_runtime_state", fail_state)
    receipt = _run_fake(backend)

    assert receipt.status == "fail"
    assert receipt.blockers[0].code == "runtime_portability_state_round_trip_failed"
    assert backend.closed_context_ids == ["fake-cpu-context"]
    assert receipt.timings.teardown_seconds >= 0.0


def test_teardown_failure_converts_passing_receipt_to_failure() -> None:
    backend = _FakePortabilityBackend(platforms=("cpu",), failure_phase="teardown")

    receipt = _run_fake(backend)

    assert receipt.status == "fail"
    assert [item.code for item in receipt.blockers] == [
        "runtime_portability_teardown_failed"
    ]
    assert receipt.result_validated
    assert receipt.synchronized
    assert receipt.runtime_state_round_trip
    assert receipt.blockers[0].details["exception_type"] == "RuntimeError"
    assert receipt.timings.teardown_seconds >= 0.0


def test_receipt_uses_teardown_timing_captured_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(range(20))
    monkeypatch.setattr(portability_module.time, "perf_counter", lambda: next(ticks))

    receipt = _run_fake(_FakePortabilityBackend(platforms=("cpu",)))

    assert receipt.status == "pass"
    assert receipt.timings.teardown_seconds == 1.0
    assert receipt.to_dict()["timings"]["teardown_seconds"] == 1.0


def test_teardown_failure_preserves_the_original_failure() -> None:
    backend = _FakePortabilityBackend(
        platforms=("cpu",),
        failure_phase="execution_and_teardown",
    )

    receipt = _run_fake(backend)

    assert receipt.status == "fail"
    assert [item.code for item in receipt.blockers] == [
        "runtime_portability_execution_failed",
        "runtime_portability_teardown_failed",
    ]
    assert backend.closed_context_ids == ["fake-cpu-context"]


def _run_fake(backend: _FakePortabilityBackend):
    return run_portability_smoke(
        "cpu",
        config=_config("cpu"),
        inspection=_inspection("cpu"),
        registry=_registry(backend),
    )


def test_portability_runtime_source_has_no_forbidden_dependencies() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "radjax_student"
        / "runtime"
        / "portability.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "radjax_student.architecture",
        "radjax_student.students",
        "radjax_student.training",
        "radjax_student.artifacts",
        "import jax",
        "import socket",
        "urllib",
        "jax.sharding",
        "Mesh(",
    ):
        assert forbidden not in source


def _registry(backend: _FakePortabilityBackend) -> RuntimeBackendRegistry:
    registry = RuntimeBackendRegistry()
    registry.register(backend)
    return registry


def _config(platform: str, *, mode: str = "eager") -> RuntimeConfig:
    return RuntimeConfig(
        backend_id="fake",
        platform_preference=platform,
        placement_policy="single_device",
        compilation_policy=mode,
        distributed_policy="disabled",
        fallback_policy="disallowed",
        seed=11,
    )


def _inspection(platform: str) -> RuntimeInspection:
    device = DeviceDescriptor(
        device_id=f"{platform}:0",
        platform=platform,
        device_kind=f"fake-{platform}",
        process_index=0,
        metadata={"jax_reported_device_id": 0},
    )
    return RuntimeInspection(
        status="pass",
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=True,
            platform=platform,
            process_count=1,
            process_index=0,
            local_device_count=1,
            global_device_count=1,
            distributed_initialized=False,
        ),
        device_inventory=DeviceInventory(
            devices=(device,),
            process_count=1,
            local_device_count=1,
            global_device_count=1,
        ),
    )


@dataclass
class _FakePortabilityBackend:
    platforms: tuple[str, ...]
    backend_id: str = "fake"
    implementation_version: str = "p2.9-test"
    notes: tuple[str, ...] = ("test-only shared portability backend",)
    placed_device_ids: list[str] = field(default_factory=list)
    closed_context_ids: list[str] = field(default_factory=list)
    compilation_modes: list[str] = field(default_factory=list)
    failure_phase: str | None = None

    @property
    def supported_platforms(self) -> tuple[str, ...]:
        return self.platforms

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            profile_id="fake.portability.v1",
            backend_id=self.backend_id,
            version=1,
            capabilities=(
                "compilation.jit_v1",
                "execution.eager_v1",
                "execution.synchronize_v1",
                "placement.single_device_v1",
                "runtime.single_process_v1",
                "state.runtime_envelope_v1",
            ),
        )

    def availability(self, inspection: RuntimeInspection) -> RuntimeBackendAvailability:
        del inspection
        return RuntimeBackendAvailability("available")

    def initialize_portability_context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection: Any,
        device: DeviceDescriptor,
    ) -> ExecutionContext:
        return ExecutionContext(
            backend_id=self.backend_id,
            environment=inspection.environment,
            device_inventory=inspection.device_inventory,
            capabilities=selection.selected_backend.capability_profile,
            root_seed=config.seed,
            runtime_id=f"fake-{device.platform}-context",
            metadata={"selected_device_id": device.device_id},
        )

    def place_portability_value(
        self,
        context: ExecutionContext,
        value: tuple[float, ...],
    ) -> _FakeVector:
        if self.failure_phase == "placement":
            raise RuntimeError("controlled placement failure")
        self.placed_device_ids.append(str(context.metadata["selected_device_id"]))
        return _FakeVector(value)

    def close_portability_context(self, context: ExecutionContext) -> None:
        self.closed_context_ids.append(context.runtime_id)
        if self.failure_phase in {"teardown", "execution_and_teardown"}:
            raise RuntimeError("controlled teardown failure")

    def prepare_runtime_execution(
        self,
        context: ExecutionContext,
        function,
        request,
        mode: str,
    ):
        del context, request
        return function, mode

    def compile_runtime_execution(self, context, handle, args, kwargs):
        del context, args, kwargs
        function, mode = handle
        if mode == "jit":
            self.compilation_modes.append(mode)
            return (function, mode), True
        return (function, mode), False

    def dispatch_runtime_execution(self, context, handle, args, kwargs):
        del context
        if self.failure_phase in {"execution", "execution_and_teardown"}:
            raise RuntimeError("controlled execution failure")
        function, _ = handle
        return function(*args, **kwargs)

    def synchronize_runtime_execution(self, context, output):
        del context
        if self.failure_phase == "synchronization":
            raise RuntimeError("controlled synchronization failure")
        return output


@dataclass(frozen=True)
class _FakeVector:
    values: tuple[float, ...]
    shape: tuple[int, ...] = (3,)
    dtype: str = "float32"

    def __mul__(self, scalar: float) -> _FakeVector:
        return _FakeVector(tuple(value * scalar for value in self.values))

    def __add__(self, scalar: float) -> _FakeVector:
        return _FakeVector(tuple(value + scalar for value in self.values))

    def __iter__(self):
        return iter(self.values)
