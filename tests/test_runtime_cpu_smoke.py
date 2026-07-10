from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from io import StringIO
from typing import Any, ClassVar

import pytest

import radjax_student.runtime.registry as registry_module
from radjax_student.cli.main import main
from radjax_student.runtime import (
    CpuRuntimeSmokeReceipt,
    DeviceDescriptor,
    DeviceInventory,
    ExecutionContext,
    JaxRuntimeBackend,
    RuntimeBackendAvailability,
    RuntimeBackendRegistry,
    RuntimeCapabilityProfile,
    RuntimeConfig,
    RuntimeEnvironment,
    RuntimeInspection,
    run_single_device_cpu_smoke,
)


def test_cpu_smoke_success_is_serializable_and_closes() -> None:
    backend = _FakeCpuBackend()

    receipt = _run(backend)
    payload = receipt.to_dict()

    assert receipt.status == "pass"
    assert receipt.backend_id == "jax"
    assert receipt.platform == "cpu"
    assert receipt.device_id == "cpu:0:0"
    assert receipt.result_validated is True
    assert receipt.synchronized is True
    assert payload["input_metadata"] == {"shape": [3], "dtype": "float32"}
    assert payload["output_metadata"] == {"shape": [3], "dtype": "float32"}
    assert backend.closed_runtime_ids == ["jax-cpu-smoke-seed-0"]
    assert CpuRuntimeSmokeReceipt.from_dict(payload) == receipt
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_concrete_jax_backend_uses_explicit_device_put_and_synchronization(
    monkeypatch,
) -> None:
    raw_device = _RawJaxCpuDevice()
    jax_module = _FakeJaxModule(raw_device)
    _FakeArray.synchronization_count = 0
    monkeypatch.setattr(
        registry_module.importlib, "import_module", lambda _: jax_module
    )
    registry = RuntimeBackendRegistry()
    registry.register(JaxRuntimeBackend())

    receipt = run_single_device_cpu_smoke(
        inspection=_inspection(),
        registry=registry,
    )

    assert receipt.status == "pass"
    assert jax_module.device_put_devices == [raw_device]
    assert jax_module.last_placed is not None
    assert _FakeArray.synchronization_count == 1


def test_cpu_smoke_selects_first_cpu_device_deterministically() -> None:
    backend = _FakeCpuBackend()

    receipt = _run(backend, _inspection(cpu_device_ids=("cpu:0:9", "cpu:0:2")))

    assert receipt.status == "pass"
    assert receipt.device_id == "cpu:0:2"
    assert backend.initialized_device_ids == ["cpu:0:2"]
    assert "runtime_multiple_cpu_devices_first_selected" in _warning_codes(receipt)


def test_cpu_smoke_rejects_non_cpu_and_multi_process_before_execution() -> None:
    backend = _FakeCpuBackend()
    non_cpu = _run(
        backend,
        config=RuntimeConfig(
            backend_id="jax",
            platform_preference="gpu",
            placement_policy="single_device",
            compilation_policy="eager",
            distributed_policy="disabled",
            fallback_policy="disallowed",
        ),
    )
    multi_process = _run(backend, _inspection(process_count=2))

    assert "runtime_cpu_platform_required" in _blocker_codes(non_cpu)
    assert "runtime_single_process_required" in _blocker_codes(multi_process)
    assert backend.initialized_device_ids == []


def test_cpu_smoke_requires_visible_cpu_device() -> None:
    backend = _FakeCpuBackend()

    receipt = _run(backend, _inspection(cpu_device_ids=()))

    assert receipt.status == "fail"
    assert "runtime_single_device_required" in _blocker_codes(receipt)
    assert backend.initialized_device_ids == []


@pytest.mark.parametrize(
    ("phase", "code"),
    [
        ("initialize", "runtime_backend_initialization_failed"),
        ("place", "runtime_placement_failed"),
        ("execute", "runtime_execution_failed"),
        ("synchronize", "runtime_synchronization_failed"),
    ],
)
def test_cpu_smoke_preserves_phase_failures_and_tears_down(
    phase: str,
    code: str,
) -> None:
    backend = _FakeCpuBackend(fail_phase=phase)

    receipt = _run(backend)

    assert receipt.status == "fail"
    assert code in _blocker_codes(receipt)
    if phase == "initialize":
        assert backend.closed_runtime_ids == []
    else:
        assert backend.closed_runtime_ids == ["jax-cpu-smoke-seed-0"]


def test_cpu_smoke_validates_result_and_preserves_teardown_failure() -> None:
    mismatch = _run(_FakeCpuBackend(result_values=(2.0, 4.0, 6.0)))
    teardown = _run(_FakeCpuBackend(fail_phase="execute", fail_teardown=True))

    assert "runtime_smoke_result_mismatch" in _blocker_codes(mismatch)
    assert "runtime_execution_failed" in _blocker_codes(teardown)
    assert "runtime_teardown_failed" in _blocker_codes(teardown)


def test_doctor_smoke_is_opt_in_and_serializes_a_receipt() -> None:
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        ("doctor", "--runtime-smoke", "--format", "json"),
        stdout=stdout,
        stderr=stderr,
    )
    payload = json.loads(stdout.getvalue())

    assert stderr.getvalue() == ""
    assert payload["runtime_smoke"] is not None
    assert code == (0 if payload["runtime_smoke"]["status"] == "pass" else 1)
    assert "CPU Runtime Smoke" in _run_doctor_human()


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="P2.4 real smoke requires the optional jax extra",
)
def test_real_jax_cpu_smoke() -> None:
    receipt = run_single_device_cpu_smoke()

    assert receipt.status == "pass"
    assert receipt.result_validated is True
    assert receipt.synchronized is True


def test_cpu_smoke_has_no_architecture_training_or_tome_imports() -> None:
    source = (
        _repo_root() / "src" / "radjax_student" / "runtime" / "smoke.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "radjax_student.artifacts",
        "radjax_student.students",
        "radjax_student.training",
        "radjax_student.losses",
        "radjax_student.schedules",
        "radjax_contract",
        "socket",
        "urllib",
    ):
        assert forbidden not in source


@dataclass
class _FakeArray:
    synchronization_count: ClassVar[int] = 0
    values: tuple[float, ...]
    shape: tuple[int, ...] = (3,)
    dtype: str = "float32"
    synchronized: bool = False

    def __mul__(self, value: int) -> _FakeArray:
        return _FakeArray(tuple(item * value for item in self.values))

    def __add__(self, value: int) -> _FakeArray:
        return _FakeArray(tuple(item + value for item in self.values))

    def block_until_ready(self) -> _FakeArray:
        self.synchronized = True
        type(self).synchronization_count += 1
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _FakeCpuBackend:
    backend_id = "jax"
    implementation_version = "test-p2.4"
    supported_platforms = ("cpu",)
    notes = ("P2.4 test double",)

    def __init__(
        self,
        *,
        fail_phase: str | None = None,
        fail_teardown: bool = False,
        result_values: tuple[float, ...] | None = None,
    ) -> None:
        self.fail_phase = fail_phase
        self.fail_teardown = fail_teardown
        self.result_values = result_values
        self.initialized_device_ids: list[str] = []
        self.closed_runtime_ids: list[str] = []

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            profile_id="jax.fake_cpu.v1",
            backend_id="jax",
            version=1,
            capabilities=(
                "placement.single_device_v1",
                "runtime.single_process_v1",
            ),
        )

    def availability(self, inspection: RuntimeInspection) -> RuntimeBackendAvailability:
        del inspection
        return RuntimeBackendAvailability("available")

    def initialize_cpu_context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection,
        device_descriptor: DeviceDescriptor,
    ) -> ExecutionContext:
        if self.fail_phase == "initialize":
            raise RuntimeError("controlled initialization failure")
        self.initialized_device_ids.append(device_descriptor.device_id)
        return ExecutionContext(
            backend_id="jax",
            environment=inspection.environment,
            device_inventory=inspection.device_inventory,
            capabilities=selection.selected_backend.capability_profile,
            root_seed=config.seed,
            runtime_id=f"jax-cpu-smoke-seed-{config.seed}",
            metadata={"selected_device_id": device_descriptor.device_id},
        )

    def place_cpu_value(self, context: ExecutionContext, value: Any) -> _FakeArray:
        del context
        if self.fail_phase == "place":
            raise RuntimeError("controlled placement failure")
        return _FakeArray(tuple(float(item) for item in value))

    def execute_cpu_smoke(
        self,
        context: ExecutionContext,
        value: _FakeArray,
    ) -> _FakeArray:
        del context
        if self.fail_phase == "execute":
            raise RuntimeError("controlled execution failure")
        if self.result_values is not None:
            return _FakeArray(self.result_values)
        return value * 2 + 1

    def synchronize_cpu_value(
        self,
        context: ExecutionContext,
        value: _FakeArray,
    ) -> _FakeArray:
        del context
        if self.fail_phase == "synchronize":
            raise RuntimeError("controlled synchronization failure")
        return value.block_until_ready()

    def close_cpu_context(self, context: ExecutionContext) -> None:
        self.closed_runtime_ids.append(context.runtime_id)
        if self.fail_teardown:
            raise RuntimeError("controlled teardown failure")


@dataclass(frozen=True)
class _RawJaxCpuDevice:
    id: int = 0
    platform: str = "cpu"
    process_index: int = 0


class _FakeJaxModule:
    def __init__(self, device: _RawJaxCpuDevice) -> None:
        self.device = device
        self.device_put_devices: list[_RawJaxCpuDevice] = []
        self.last_placed: _FakeArray | None = None

    def devices(self, platform: str) -> tuple[_RawJaxCpuDevice, ...]:
        assert platform == "cpu"
        return (self.device,)

    def device_put(
        self,
        value: tuple[float, ...],
        device: _RawJaxCpuDevice,
    ) -> _FakeArray:
        self.device_put_devices.append(device)
        self.last_placed = _FakeArray(tuple(float(item) for item in value))
        return self.last_placed


def _run(
    backend: _FakeCpuBackend,
    inspection: RuntimeInspection | None = None,
    *,
    config: RuntimeConfig | None = None,
):
    registry = RuntimeBackendRegistry()
    registry.register(backend)
    return run_single_device_cpu_smoke(
        config=config,
        inspection=_inspection() if inspection is None else inspection,
        registry=registry,
    )


def _inspection(
    *,
    process_count: int = 1,
    cpu_device_ids: tuple[str, ...] = ("cpu:0:0",),
) -> RuntimeInspection:
    devices = tuple(
        DeviceDescriptor(
            device_id=device_id,
            platform="cpu",
            process_index=0,
            metadata={"jax_reported_device_id": index},
        )
        for index, device_id in enumerate(cpu_device_ids)
    )
    return RuntimeInspection(
        status="pass",
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=True,
            platform="cpu",
            process_count=process_count,
            process_index=0,
            local_device_count=len(devices),
            global_device_count=len(devices),
            distributed_initialized=False,
        ),
        device_inventory=DeviceInventory(
            devices=devices,
            process_count=process_count,
            local_device_count=len(devices),
            global_device_count=len(devices),
        ),
    )


def _blocker_codes(receipt) -> list[str]:
    return [item.code for item in receipt.blockers]


def _warning_codes(receipt) -> list[str]:
    return [item.code for item in receipt.warnings]


def _run_doctor_human() -> str:
    stdout = StringIO()
    code = main(("doctor",), stdout=stdout, stderr=StringIO())
    assert code == 0
    return stdout.getvalue()


def _repo_root():
    return __import__("pathlib").Path(__file__).resolve().parents[1]
