from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from radjax_student.runtime import (
    COMPILATION_POLICIES,
    DISTRIBUTED_POLICIES,
    FALLBACK_POLICIES,
    PLACEMENT_POLICIES,
    PRECISION_POLICIES,
    RUNTIME_CAPABILITY_VOCABULARY,
    RUNTIME_ERROR_CODES,
    CompilationOptions,
    DeviceDescriptor,
    DeviceInventory,
    ExecutionContext,
    RuntimeBackend,
    RuntimeCapabilityProfile,
    RuntimeConfig,
    RuntimeContractError,
    RuntimeEnvironment,
    RuntimeIssue,
    RuntimeReport,
    RuntimeState,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_contract_import_does_not_initialize_optional_ml_stacks() -> None:
    script = """
import builtins
import sys
real_import = builtins.__import__
forbidden = {
    "jax", "jaxlib", "torch", "transformers", "datasets", "accelerate",
    "radjax_tome",
}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from radjax_student.runtime import RuntimeConfig, RuntimeEnvironment
assert RuntimeConfig().backend_id is None
assert RuntimeEnvironment(python_version=sys.version.split()[0], jax_available=False)
assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_config_round_trips_requested_policy() -> None:
    config = RuntimeConfig(
        backend_id="fake",
        platform_preference="cpu",
        precision_policy="float32",
        placement_policy="single_device",
        compilation_policy="eager",
        distributed_policy="disabled",
        fallback_policy="disallowed",
        required_capabilities=(
            "runtime.single_process_v1",
            "placement.single_device_v1",
        ),
        seed=41,
        debug=True,
    )

    assert RuntimeConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["required_capabilities"] == [
        "placement.single_device_v1",
        "runtime.single_process_v1",
    ]
    assert "architecture" not in config.to_dict()
    json.dumps(config.to_dict())
    with pytest.raises(TypeError, match="debug must be a boolean"):
        RuntimeConfig.from_dict({"debug": "false"})


def test_requested_policy_and_observed_environment_remain_distinct() -> None:
    config = RuntimeConfig(
        platform_preference="gpu",
        distributed_policy="required",
    )
    environment = RuntimeEnvironment(
        python_version="3.11.9",
        jax_available=False,
        platform=None,
        process_count=None,
        distributed_initialized=None,
        warnings=("inspection_deferred",),
    )

    assert config.platform_preference == "gpu"
    assert config.distributed_policy == "required"
    assert environment.platform is None
    assert environment.distributed_initialized is None
    assert RuntimeEnvironment.from_dict(environment.to_dict()) == environment


def test_device_models_preserve_unknown_values_without_raw_backend_objects() -> None:
    descriptor = DeviceDescriptor(
        device_id="unknown:0",
        platform=None,
        device_kind=None,
        process_index=None,
        local_hardware_id=None,
        memory_bytes=None,
        metadata={"vendor_extension": {"known": False}},
    )
    inventory = DeviceInventory(
        devices=(descriptor,),
        process_count=None,
        local_device_count=None,
        global_device_count=None,
        topology_summary={"mesh": None},
    )

    assert DeviceDescriptor.from_dict(descriptor.to_dict()) == descriptor
    assert DeviceInventory.from_dict(inventory.to_dict()) == inventory
    assert inventory.devices[0].platform is None
    assert inventory.topology_summary["mesh"] is None
    with pytest.raises(TypeError):
        descriptor.metadata["changed"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        DeviceDescriptor(device_id="bad", metadata={"raw": object()})
    with pytest.raises(TypeError, match="local_hardware_id"):
        DeviceDescriptor(device_id="bad", local_hardware_id=object())


def test_capability_profile_is_versioned_deterministic_and_honest() -> None:
    profile = RuntimeCapabilityProfile(
        profile_id="fake.runtime.v1",
        backend_id="fake",
        version=1,
        capabilities=(
            "runtime.single_process_v1",
            "placement.single_device_v1",
        ),
        non_capabilities=("compilation.jit_v1",),
        notes=("declaration only",),
    )

    assert profile.capabilities == (
        "placement.single_device_v1",
        "runtime.single_process_v1",
    )
    assert RuntimeCapabilityProfile.from_dict(profile.to_dict()) == profile
    assert json.dumps(profile.to_dict(), sort_keys=True) == json.dumps(
        profile.to_dict(),
        sort_keys=True,
    )
    with pytest.raises(ValueError, match="overlap"):
        RuntimeCapabilityProfile(
            profile_id="bad",
            backend_id="fake",
            version=1,
            capabilities=("runtime.single_process_v1",),
            non_capabilities=("runtime.single_process_v1",),
        )


def test_policy_and_capability_vocabularies_are_frozen() -> None:
    assert PRECISION_POLICIES == (
        "float32",
        "bfloat16",
        "float16",
        "mixed",
        "automatic",
        "unspecified",
    )
    assert PLACEMENT_POLICIES == (
        "single_device",
        "replicated",
        "data_sharded",
        "model_sharded",
        "automatic",
        "unspecified",
    )
    assert COMPILATION_POLICIES == ("eager", "jit", "automatic", "unspecified")
    assert DISTRIBUTED_POLICIES == ("disabled", "auto", "required")
    assert FALLBACK_POLICIES == ("disallowed", "allow_compatible")
    assert RUNTIME_CAPABILITY_VOCABULARY == tuple(sorted(RUNTIME_CAPABILITY_VOCABULARY))


def test_execution_context_is_runtime_state_not_model_state() -> None:
    context = _context()
    payload = context.to_dict()

    assert ExecutionContext.from_dict(payload) == context
    assert payload["root_seed"] == 7
    assert "params" not in payload
    assert "model" not in payload
    assert "optimizer" not in payload
    assert "rng" not in context.__dict__
    json.dumps(payload)


def test_runtime_state_envelope_round_trips_without_persistence_behavior() -> None:
    state = RuntimeState(
        runtime_id="runtime-test-1",
        global_step=12,
        root_seed=7,
        runtime_config=RuntimeConfig(seed=7),
        topology_summary={"process_count": 1},
        precision_policy="float32",
        placement_policy="single_device",
        resume_metadata={"source": "test"},
    )

    assert RuntimeState.from_dict(state.to_dict()) == state
    assert "model_parameters" not in state.to_dict()
    assert "optimizer_state" not in state.to_dict()
    assert not hasattr(state, "save")
    assert not hasattr(state, "restore")


def test_compilation_options_are_narrow_and_serializable() -> None:
    options = CompilationOptions(
        enabled=True,
        static_arg_names=("mode",),
        donate_arg_names=("buffer",),
        debug=True,
        synchronize_results=True,
    )

    assert CompilationOptions.from_dict(options.to_dict()) == options
    assert set(options.to_dict()) == {
        "enabled",
        "static_arg_names",
        "donate_arg_names",
        "debug",
        "synchronize_results",
    }


def test_structured_runtime_errors_preserve_stable_code_and_details() -> None:
    error = RuntimeContractError(
        "runtime_fallback_disallowed",
        "requested GPU is unavailable and fallback is disabled",
        details={"requested_platform": "gpu", "available_platforms": ["cpu"]},
    )

    assert error.code == "runtime_fallback_disallowed"
    assert error.code in RUNTIME_ERROR_CODES
    assert error.to_dict() == {
        "code": "runtime_fallback_disallowed",
        "message": "requested GPU is unavailable and fallback is disabled",
        "details": {
            "requested_platform": "gpu",
            "available_platforms": ["cpu"],
        },
    }
    assert "runtime_fallback_disallowed" in str(error)
    with pytest.raises(TypeError):
        error.details["requested_platform"] = "cpu"  # type: ignore[index]


def test_runtime_report_round_trips_to_valid_json() -> None:
    context = _context()
    report = RuntimeReport(
        status="fail",
        backend_id="fake",
        environment=context.environment,
        device_inventory=context.device_inventory,
        capabilities=context.capabilities,
        selected_policy=RuntimeConfig(
            backend_id="fake",
            platform_preference="gpu",
            fallback_policy="disallowed",
        ),
        blockers=(
            RuntimeIssue.create(
                "requested_platform_unavailable",
                "requested platform was not observed",
                requested_platform="gpu",
            ),
        ),
        warnings=(
            RuntimeIssue.create(
                "runtime_environment_partial",
                "full inspection is deferred to P2.2",
            ),
        ),
        claims_not_made=(
            "backend_not_selected",
            "execution_not_attempted",
            "model_quality_not_claimed",
        ),
    )
    payload = report.to_dict()

    assert not report.ok
    assert RuntimeReport.from_dict(payload) == report
    assert json.loads(json.dumps(payload)) == payload
    assert payload["blockers"][0]["code"] == "requested_platform_unavailable"
    assert payload["claims_not_made"][-1] == "model_quality_not_claimed"

    with pytest.raises(ValueError, match="unknown blocker codes"):
        RuntimeReport(
            status="fail",
            backend_id=None,
            environment=None,
            device_inventory=None,
            capabilities=None,
            selected_policy=None,
            blockers=(RuntimeIssue.create("vague_failure", "failed"),),
        )


def test_fake_backend_satisfies_protocol_without_real_execution() -> None:
    backend = _FakeBackend()

    assert isinstance(backend, RuntimeBackend)
    assert backend.backend_id == "fake"
    assert backend.capability_profile().capabilities == ("runtime.single_process_v1",)


def test_generic_runtime_modules_do_not_import_product_or_policy_layers() -> None:
    runtime_root = REPO_ROOT / "src" / "radjax_student" / "runtime"
    forbidden = (
        "radjax_student.architecture",
        "radjax_student.artifacts",
        "radjax_student.losses",
        "radjax_student.schedules",
        "radjax_student.students",
        "radjax_student.training",
        "radjax_contract",
        "radjax_tome",
        "jax",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
    )
    offenders: list[str] = []
    for path in runtime_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for dependency in forbidden:
            if f"import {dependency}" in source or f"from {dependency}" in source:
                offenders.append(f"{path.name}: {dependency}")

    assert offenders == []


class _FakeBackend:
    @property
    def backend_id(self) -> str:
        return "fake"

    def inspect_environment(self) -> RuntimeEnvironment:
        return _environment()

    def capability_profile(self) -> RuntimeCapabilityProfile:
        return RuntimeCapabilityProfile(
            profile_id="fake.runtime.v1",
            backend_id="fake",
            version=1,
            capabilities=("runtime.single_process_v1",),
            non_capabilities=("compilation.jit_v1",),
        )

    def initialize(self, config: RuntimeConfig) -> ExecutionContext:
        del config
        return _context()

    def place(self, value: Any, placement: str) -> Any:
        del placement
        return value

    def compile(
        self,
        function,
        options: CompilationOptions,
    ):
        del options
        return function

    def synchronize(self, value: Any) -> Any:
        return value

    def close(self, context: ExecutionContext) -> None:
        del context


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.11.9",
        jax_available=False,
        platform="cpu",
        process_count=1,
        process_index=0,
        local_device_count=1,
        global_device_count=1,
        distributed_initialized=False,
    )


def _context() -> ExecutionContext:
    environment = _environment()
    inventory = DeviceInventory(
        devices=(
            DeviceDescriptor(
                device_id="fake:0",
                platform="cpu",
                device_kind="fake",
                process_index=0,
                local_hardware_id=0,
                supported_precisions=("float32",),
            ),
        ),
        process_count=1,
        local_device_count=1,
        global_device_count=1,
        topology_summary={"kind": "single_process"},
    )
    capabilities = RuntimeCapabilityProfile(
        profile_id="fake.runtime.v1",
        backend_id="fake",
        version=1,
        capabilities=("runtime.single_process_v1",),
        non_capabilities=("compilation.jit_v1",),
    )
    return ExecutionContext(
        backend_id="fake",
        environment=environment,
        device_inventory=inventory,
        capabilities=capabilities,
        root_seed=7,
        runtime_id="runtime-test-1",
        metadata={"test_double": True},
    )
