from __future__ import annotations

import json
import socket
import urllib.request
from io import StringIO
from types import SimpleNamespace

import pytest

import radjax_student.reports.doctor as doctor_module
import radjax_student.runtime.inspection as inspection_module
from radjax_student.cli.main import main
from radjax_student.runtime import (
    DeviceInventory,
    RuntimeEnvironment,
    RuntimeInspection,
    RuntimeIssue,
    inspect_runtime_environment,
)


def test_jax_absence_is_a_healthy_observed_fact(monkeypatch) -> None:
    monkeypatch.setattr(inspection_module, "_find_module_spec", lambda name: None)

    inspection = inspect_runtime_environment()

    assert inspection.status == "pass"
    assert inspection.environment.jax_available is False
    assert inspection.environment.jax_version is None
    assert inspection.environment.jaxlib_version is None
    assert inspection.environment.platform is None
    assert inspection.environment.process_count is None
    assert inspection.environment.process_index is None
    assert inspection.environment.local_device_count is None
    assert inspection.environment.global_device_count is None
    assert inspection.device_inventory.devices == ()
    assert _warning_codes(inspection) == ["jax_not_installed"]
    assert inspection.blockers == ()


def test_installed_but_broken_jax_is_distinct_from_absence(monkeypatch) -> None:
    monkeypatch.setattr(
        inspection_module,
        "_find_module_spec",
        lambda name: object(),
    )

    def broken_import(name: str):
        raise RuntimeError(
            f"controlled {name} import failure at 0xABC in /tmp/runtime-probe"
        )

    monkeypatch.setattr(inspection_module, "_import_module", broken_import)

    inspection = inspect_runtime_environment()

    assert inspection.status == "pass"
    assert inspection.environment.jax_available is False
    assert _warning_codes(inspection) == ["jax_import_failed"]
    details = inspection.warnings[0].details
    assert details["exception_type"] == "RuntimeError"
    assert "controlled jax import failure" in details["exception_message"]
    assert "<memory-address>" in details["exception_message"]
    assert "<path>" in details["exception_message"]
    assert "0xABC" not in details["exception_message"]
    assert "/tmp/runtime-probe" not in details["exception_message"]


def test_jaxlib_import_failure_preserves_partial_jax_facts(monkeypatch) -> None:
    jax_module = _fake_jax((_FakeDevice(0, "cpu", "Fake CPU", 0, 0),))

    def importer(name: str):
        if name == "jax":
            return jax_module
        raise ImportError("controlled jaxlib failure")

    _install_fake_imports(monkeypatch, importer)

    inspection = inspect_runtime_environment()

    assert inspection.status == "pass"
    assert inspection.environment.jax_available is True
    assert inspection.environment.jax_version == "0.test"
    assert inspection.environment.jaxlib_version is None
    assert "jaxlib_import_failed" in _warning_codes(inspection)


def test_one_cpu_device_is_normalized_without_raw_object(monkeypatch) -> None:
    raw_device = _FakeDevice(7, "cpu", "Fake CPU", 0, 3)
    _install_healthy_fake_jax(monkeypatch, (raw_device,))

    inspection = inspect_runtime_environment()
    payload = inspection.to_dict()
    device = inspection.device_inventory.devices[0]

    assert inspection.status == "pass"
    assert inspection.environment.platform == "cpu"
    assert inspection.environment.process_count == 1
    assert inspection.environment.process_index == 0
    assert inspection.environment.local_device_count == 1
    assert inspection.environment.global_device_count == 1
    assert inspection.environment.distributed_initialized is False
    assert device.device_id == "cpu:0:0"
    assert device.platform == "cpu"
    assert device.device_kind == "Fake CPU"
    assert device.local_hardware_id == 3
    assert device.memory_bytes is None
    assert device.supported_precisions == ()
    assert device.metadata == {"jax_reported_device_id": 7}
    assert raw_device not in payload["device_inventory"]["devices"]
    assert "device_memory_unknown" in _warning_codes(inspection)
    assert "device_precision_unknown" in _warning_codes(inspection)


def test_multi_device_ids_are_unique_and_deterministic(monkeypatch) -> None:
    devices = (
        _FakeDevice(2, "cpu", "Fake CPU", 0, 0),
        _FakeDevice(3, "cpu", "Fake CPU", 0, 1),
        _FakeDevice(4, "cpu", "Fake CPU", 0, 2),
    )
    _install_healthy_fake_jax(monkeypatch, devices)

    first = inspect_runtime_environment()
    second = inspect_runtime_environment()
    identifiers = tuple(item.device_id for item in first.device_inventory.devices)

    assert identifiers == ("cpu:0:0", "cpu:0:1", "cpu:0:2")
    assert len(set(identifiers)) == 3
    assert first.to_dict() == second.to_dict()


def test_heterogeneous_platforms_are_preserved_and_warned(monkeypatch) -> None:
    devices = (
        _FakeDevice(0, "cpu", "Fake CPU", 0, 0),
        _FakeDevice(1, "gpu", "Fake GPU", 0, 1),
    )
    _install_healthy_fake_jax(monkeypatch, devices, default_platform="gpu")

    inspection = inspect_runtime_environment()

    assert inspection.status == "pass"
    assert [item.platform for item in inspection.device_inventory.devices] == [
        "cpu",
        "gpu",
    ]
    assert inspection.device_inventory.topology_summary["platforms"] == (
        "cpu",
        "gpu",
    )
    assert "heterogeneous_platforms_detected" in _warning_codes(inspection)


@pytest.mark.parametrize(
    ("process_count", "process_index", "global_count"),
    [
        (1, 1, 1),
        (1, 0, 2),
        (-1, 0, 1),
    ],
)
def test_inconsistent_counts_fail_with_structured_blocker(
    monkeypatch,
    process_count: int,
    process_index: int,
    global_count: int,
) -> None:
    devices = (_FakeDevice(0, "cpu", "Fake CPU", 0, 0),)
    jax_module = _fake_jax(
        devices,
        process_count=process_count,
        process_index=process_index,
        global_count=global_count,
    )
    _install_fake_imports(monkeypatch, _module_importer(jax_module))

    inspection = inspect_runtime_environment()

    assert inspection.status == "fail"
    assert inspection.blockers
    assert all(
        blocker.code == "device_normalization_failed" for blocker in inspection.blockers
    )


def test_inspection_json_round_trip_is_deterministic(monkeypatch) -> None:
    _install_healthy_fake_jax(
        monkeypatch,
        (_FakeDevice(0, "cpu", "Fake CPU", 0, 0),),
    )

    inspection = inspect_runtime_environment()
    payload = inspection.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert RuntimeInspection.from_dict(payload) == inspection
    assert json.loads(encoded) == payload
    assert "0x" not in encoded
    assert "Traceback" not in encoded


def test_inspection_does_not_execute_jax_operations_or_network(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("execution-only operation was called")

    device = _FakeDevice(0, "cpu", "Fake CPU", 0, 0)
    device.block_until_ready = forbidden
    jax_module = _fake_jax((device,))
    jax_module.jit = forbidden
    jax_module.device_put = forbidden
    jax_module.numpy = SimpleNamespace(array=forbidden)
    _install_fake_imports(monkeypatch, _module_importer(jax_module))
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    inspection = inspect_runtime_environment()

    assert inspection.status == "pass"
    assert not hasattr(inspection, "backend")
    assert not hasattr(inspection, "context")
    assert not hasattr(inspection, "runtime_id")


def test_doctor_reports_absent_jax_as_healthy_runtime_inspection(
    monkeypatch,
) -> None:
    absent = _absent_inspection()
    monkeypatch.setattr(doctor_module, "inspect_runtime_environment", lambda: absent)
    stdout = StringIO()
    stderr = StringIO()

    code = main(("doctor", "--format", "json"), stdout=stdout, stderr=stderr)
    payload = json.loads(stdout.getvalue())

    assert code == 0
    assert stderr.getvalue() == ""
    assert payload["runtime_inspection"]["status"] == "pass"
    assert payload["runtime_inspection"]["environment"]["jax_available"] is False
    assert payload["capability_state"]["runtime_inspection"] == "available"
    assert payload["capability_state"]["runtime_backend_registry"] == "available"
    assert payload["capability_state"]["runtime_backend_selection"] == "available"
    assert (
        payload["capability_state"]["runtime_cpu_smoke"]
        == "available_on_explicit_request"
    )
    assert payload["capability_state"]["placement_intent"] == "available"
    assert payload["capability_state"]["execution_boundary"] == "available"
    assert (
        payload["capability_state"]["runtime_state"] == "available_on_explicit_request"
    )
    assert payload["placement_intent"]["concrete_resolution"] == [
        "single_device_cpu_smoke_only"
    ]
    assert (
        payload["execution_boundary"]["automatic"] == "resolves_to_eager_with_warning"
    )
    assert payload["capability_state"]["jax_execution"] == "unavailable"
    assert payload["runtime_backend_descriptors"][0]["backend_id"] == "jax"
    assert payload["runtime_selection"]["status"] == "fail"

    human_stdout = StringIO()
    human_code = main(("doctor",), stdout=human_stdout, stderr=StringIO())
    assert human_code == 0
    assert "Runtime Inspection" in human_stdout.getvalue()
    assert "Runtime Backend Selection" in human_stdout.getvalue()
    assert "Placement Intent" in human_stdout.getvalue()
    assert "Execution Boundary" in human_stdout.getvalue()
    assert "Runtime State Smoke" in human_stdout.getvalue()
    assert "JAX available: no" in human_stdout.getvalue()
    assert "warnings: jax_not_installed" in human_stdout.getvalue()
    assert "JAX execution: unavailable" in human_stdout.getvalue()


def test_doctor_fails_only_when_runtime_inspection_is_incoherent(monkeypatch) -> None:
    failed = RuntimeInspection(
        status="fail",
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=False,
            warnings=("runtime_inspection_internal_error",),
        ),
        device_inventory=DeviceInventory(),
        blockers=(
            RuntimeIssue.create(
                "runtime_inspection_internal_error",
                "controlled inspection failure",
            ),
        ),
    )
    monkeypatch.setattr(doctor_module, "inspect_runtime_environment", lambda: failed)
    stdout = StringIO()
    stderr = StringIO()

    code = main(("doctor",), stdout=stdout, stderr=stderr)

    assert code == 1
    assert stderr.getvalue() == ""
    assert "Status: FAIL" in stdout.getvalue()
    assert "runtime_inspection_failed" in stdout.getvalue()


def _absent_inspection() -> RuntimeInspection:
    warning = RuntimeIssue.create(
        "jax_not_installed",
        "JAX is not installed",
    )
    return RuntimeInspection(
        status="pass",
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=False,
            warnings=(warning.code,),
        ),
        device_inventory=DeviceInventory(),
        warnings=(warning,),
    )


def _install_healthy_fake_jax(
    monkeypatch,
    devices: tuple[_FakeDevice, ...],
    *,
    default_platform: str = "cpu",
) -> None:
    jax_module = _fake_jax(devices, default_platform=default_platform)
    _install_fake_imports(monkeypatch, _module_importer(jax_module))


def _install_fake_imports(monkeypatch, importer) -> None:
    monkeypatch.setattr(
        inspection_module,
        "_find_module_spec",
        lambda name: object(),
    )
    monkeypatch.setattr(inspection_module, "_import_module", importer)


def _module_importer(jax_module):
    jaxlib_module = SimpleNamespace(__name__="jaxlib", __version__="0.test")

    def importer(name: str):
        return jax_module if name == "jax" else jaxlib_module

    return importer


def _fake_jax(
    devices: tuple[_FakeDevice, ...],
    *,
    default_platform: str = "cpu",
    process_count: int = 1,
    process_index: int = 0,
    global_count: int | None = None,
):
    count = len(devices) if global_count is None else global_count
    return SimpleNamespace(
        __name__="jax",
        __version__="0.test",
        default_backend=lambda: default_platform,
        process_count=lambda: process_count,
        process_index=lambda: process_index,
        local_device_count=lambda: len(devices),
        device_count=lambda: count,
        devices=lambda: devices,
        distributed=SimpleNamespace(is_initialized=lambda: False),
    )


class _FakeDevice:
    def __init__(
        self,
        device_id: int,
        platform: str,
        device_kind: str,
        process_index: int,
        local_hardware_id: int,
    ) -> None:
        self.id = device_id
        self.platform = platform
        self.device_kind = device_kind
        self.process_index = process_index
        self.local_hardware_id = local_hardware_id


def _warning_codes(inspection: RuntimeInspection) -> list[str]:
    return [warning.code for warning in inspection.warnings]
