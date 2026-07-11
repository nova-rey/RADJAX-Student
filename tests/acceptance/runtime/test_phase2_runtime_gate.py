from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

from radjax_student.cli.main import main
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
    default_cpu_smoke_config,
    run_portability_smoke,
    run_selected_cpu_smoke,
    select_runtime_backend,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = REPO_ROOT / "runtime_phase2_acceptance_receipt.json"
EXPECTED_GATES = {
    "runtime_contract_and_import_boundary",
    "environment_and_backend_selection",
    "cpu_heartbeat",
    "rng_and_runtime_state",
    "placement_and_execution_boundary",
    "portability_and_teardown_receipt",
    "doctor_integration",
}
EXPECTED_NON_CLAIMS = {
    "training_not_run",
    "gradients_not_computed",
    "optimizer_not_updated",
    "model_initialization_not_tested",
    "tome_payload_loading_not_tested",
    "distributed_execution_not_tested",
    "sharding_not_tested",
    "scale_not_tested",
    "performance_not_benchmarked",
    "model_quality_not_claimed",
}


def test_phase2_receipt_is_complete_deterministic_and_self_consistent() -> None:
    text = RECEIPT_PATH.read_text(encoding="utf-8")
    receipt = json.loads(text)

    assert text == json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    assert receipt["gate"] == "P2.10"
    assert receipt["gate_version"] == 1
    assert receipt["status"] == "pass"
    assert receipt["blockers"] == []
    assert receipt["runtime"]["accepted_runtime_commit"] == (
        "e3aceebf97fab8a900a9c712e86c41ae5c72871b"
    )
    assert receipt["runtime"]["schema_versions"] == {
        "execution_capabilities": "execution_capabilities.v1",
        "placement_capabilities": "placement_capabilities.v1",
        "runtime_keys": "runtime_keys.v1",
        "runtime_state": "runtime_state.v1",
    }
    assert set(receipt["passing_gates"]) == EXPECTED_GATES
    assert receipt["optional_accelerator_receipts"] == {
        "gpu": "external_when_available",
        "tpu": "external_when_available",
    }
    assert set(receipt["claims_not_made"]) == EXPECTED_NON_CLAIMS
    assert receipt["phase_status"] == {
        "phase_2_student_runtime": "complete",
        "phase_3_generic_training_core": "unblocked",
    }
    assert receipt["required_ci"] == [
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q "
        "tests/acceptance/runtime",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q",
    ]


def test_phase2_gate_runs_one_complete_fake_cpu_runtime_trace() -> None:
    backend = _GateBackend()
    registry = RuntimeBackendRegistry()
    registry.register(backend)
    inspection = _inspection()
    config = default_cpu_smoke_config()
    selection = select_runtime_backend(config, inspection, registry)

    heartbeat = run_selected_cpu_smoke(
        config=config,
        inspection=inspection,
        selection=selection,
        backend=backend,
    )
    portability = run_portability_smoke(
        "cpu",
        config=config,
        inspection=inspection,
        registry=registry,
    )

    assert heartbeat.ok
    assert heartbeat.result_validated
    assert heartbeat.synchronized
    assert portability.ok
    assert portability.result_validated
    assert portability.synchronized
    assert portability.runtime_state_round_trip
    assert portability.timings.teardown_seconds >= 0.0
    assert backend.cpu_context_closes == ["gate-cpu-context"]
    assert backend.portability_context_closes == ["gate-cpu-portability"]
    assert backend.placed_device_ids == ["cpu:0", "cpu:0"]
    assert backend.execution_dispatches == 1


def test_default_doctor_includes_runtime_surfaces_without_execution() -> None:
    stdout = StringIO()
    stderr = StringIO()

    code = main(("doctor", "--format", "json"), stdout=stdout, stderr=stderr)
    payload = json.loads(stdout.getvalue())

    assert code == 0
    assert stderr.getvalue() == ""
    assert payload["runtime_smoke"] is None
    assert payload["runtime_state_smoke"] is None
    assert payload["runtime_portability_smoke"] is None
    assert payload["capability_state"]["runtime_portability"] == (
        "available_on_explicit_request"
    )


def test_gate_sources_remain_architecture_and_training_independent() -> None:
    runtime_root = REPO_ROOT / "src" / "radjax_student" / "runtime"
    sources = (
        runtime_root / "inspection.py",
        runtime_root / "selection.py",
        runtime_root / "execution.py",
        runtime_root / "state.py",
        runtime_root / "portability.py",
    )

    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        for forbidden in (
            "radjax_student.architecture",
            "radjax_student.students",
            "radjax_student.training",
            "radjax_student.artifacts",
            "jax.sharding",
            "Mesh(",
        ):
            assert forbidden not in source, (source_path, forbidden)


def _inspection() -> RuntimeInspection:
    device = DeviceDescriptor(
        device_id="cpu:0",
        platform="cpu",
        device_kind="gate-cpu",
        process_index=0,
        metadata={"jax_reported_device_id": 0},
    )
    return RuntimeInspection(
        status="pass",
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=True,
            jax_version="0.gate",
            jaxlib_version="0.gate",
            platform="cpu",
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
class _GateBackend:
    backend_id: str = "jax"
    implementation_version: str = "p2.10-gate"
    supported_platforms: tuple[str, ...] = ("cpu",)
    notes: tuple[str, ...] = ("deterministic P2.10 acceptance backend",)
    placed_device_ids: list[str] = field(default_factory=list)
    cpu_context_closes: list[str] = field(default_factory=list)
    portability_context_closes: list[str] = field(default_factory=list)
    execution_dispatches: int = 0

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            profile_id="jax.runtime.p2.10-gate",
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

    def initialize_cpu_context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection: Any,
        device: DeviceDescriptor,
    ) -> ExecutionContext:
        return self._context(
            config,
            inspection,
            selection,
            device,
            runtime_id="gate-cpu-context",
        )

    def place_cpu_value(
        self,
        context: ExecutionContext,
        value: tuple[float, ...],
    ) -> _GateVector:
        self.placed_device_ids.append(str(context.metadata["selected_device_id"]))
        return _GateVector(value)

    def execute_cpu_smoke(
        self, context: ExecutionContext, value: _GateVector
    ) -> _GateVector:
        del context
        return value * 2 + 1

    def synchronize_cpu_value(
        self,
        context: ExecutionContext,
        value: _GateVector,
    ) -> _GateVector:
        del context
        return value

    def close_cpu_context(self, context: ExecutionContext) -> None:
        self.cpu_context_closes.append(context.runtime_id)

    def initialize_portability_context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection: Any,
        device: DeviceDescriptor,
    ) -> ExecutionContext:
        return self._context(
            config,
            inspection,
            selection,
            device,
            runtime_id="gate-cpu-portability",
        )

    def place_portability_value(
        self,
        context: ExecutionContext,
        value: tuple[float, ...],
    ) -> _GateVector:
        self.placed_device_ids.append(str(context.metadata["selected_device_id"]))
        return _GateVector(value)

    def close_portability_context(self, context: ExecutionContext) -> None:
        self.portability_context_closes.append(context.runtime_id)

    def prepare_runtime_execution(self, context, function, request, mode):
        del context, request
        return function, mode

    def compile_runtime_execution(self, context, handle, args, kwargs):
        del context, args, kwargs
        return handle, False

    def dispatch_runtime_execution(self, context, handle, args, kwargs):
        del context
        self.execution_dispatches += 1
        function, _ = handle
        return function(*args, **kwargs)

    def synchronize_runtime_execution(self, context, output):
        del context
        return output

    def _context(
        self,
        config: RuntimeConfig,
        inspection: RuntimeInspection,
        selection: Any,
        device: DeviceDescriptor,
        *,
        runtime_id: str,
    ) -> ExecutionContext:
        return ExecutionContext(
            backend_id=self.backend_id,
            environment=inspection.environment,
            device_inventory=inspection.device_inventory,
            capabilities=selection.selected_backend.capability_profile,
            root_seed=config.seed,
            runtime_id=runtime_id,
            metadata={"selected_device_id": device.device_id},
        )


@dataclass(frozen=True)
class _GateVector:
    values: tuple[float, ...]
    shape: tuple[int, ...] = (3,)
    dtype: str = "float32"

    def __mul__(self, scalar: float) -> _GateVector:
        return _GateVector(tuple(value * scalar for value in self.values))

    def __add__(self, scalar: float) -> _GateVector:
        return _GateVector(tuple(value + scalar for value in self.values))

    def __iter__(self):
        return iter(self.values)

    def tolist(self) -> list[float]:
        return list(self.values)
