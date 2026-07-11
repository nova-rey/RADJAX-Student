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
        self.placed_device_ids.append(str(context.metadata["selected_device_id"]))
        return _FakeVector(value)

    def close_portability_context(self, context: ExecutionContext) -> None:
        self.closed_context_ids.append(context.runtime_id)

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
        function, _ = handle
        return function(*args, **kwargs)

    def synchronize_runtime_execution(self, context, output):
        del context
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
