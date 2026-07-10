from __future__ import annotations

import json

import pytest

from radjax_student.runtime import (
    DeviceDescriptor,
    DeviceInventory,
    FakeRuntimeBackend,
    RuntimeBackendRegistry,
    RuntimeConfig,
    RuntimeContractError,
    RuntimeEnvironment,
    RuntimeInspection,
    RuntimeSelectionResult,
    build_default_runtime_registry,
    select_runtime_backend,
)


def test_registry_rejects_duplicates_and_lists_backends_deterministically() -> None:
    registry = RuntimeBackendRegistry()
    registry.register(FakeRuntimeBackend(backend_id="zeta"))
    registry.register(FakeRuntimeBackend(backend_id="alpha"))

    assert [item.backend_id for item in registry.list_backends()] == ["alpha", "zeta"]
    with pytest.raises(RuntimeContractError, match="runtime_backend_duplicate"):
        registry.register(FakeRuntimeBackend(backend_id="alpha"))

    removed = registry.unregister("alpha")
    assert removed.backend_id == "alpha"
    assert not registry.contains("alpha")


def test_default_registry_is_safe_without_jax_and_excludes_fake_backend() -> None:
    registry = build_default_runtime_registry()

    assert [item.backend_id for item in registry.list_backends()] == ["jax"]
    descriptor = registry.describe(_inspection(jax_available=False))[0]
    assert descriptor.backend_id == "jax"
    assert descriptor.availability.status == "unavailable"
    assert "fake" not in [item.backend_id for item in registry.list_backends()]


def test_explicit_fake_selection_and_json_round_trip() -> None:
    registry = RuntimeBackendRegistry()
    registry.register(
        FakeRuntimeBackend(
            backend_id="cpu-test",
            capabilities=("placement.single_device_v1", "required.v1"),
        )
    )
    config = RuntimeConfig(
        backend_id="cpu-test",
        platform_preference="cpu",
        required_capabilities=("required.v1",),
    )

    selection = select_runtime_backend(config, _inspection(), registry)
    payload = selection.to_dict()

    assert selection.status == "pass"
    assert selection.selected_backend is not None
    assert selection.selected_backend.backend_id == "cpu-test"
    assert selection.selected_platform == "cpu"
    assert selection.satisfied_capabilities == ("required.v1",)
    assert RuntimeSelectionResult.from_dict(payload) == selection
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_selection_reports_missing_backend_unavailability_and_capability() -> None:
    available = RuntimeBackendRegistry()
    available.register(FakeRuntimeBackend(backend_id="available"))
    unavailable = RuntimeBackendRegistry()
    unavailable.register(FakeRuntimeBackend(backend_id="offline", available=False))

    missing_backend = select_runtime_backend(
        RuntimeConfig(backend_id="unknown"),
        _inspection(),
        available,
    )
    unavailable_backend = select_runtime_backend(
        RuntimeConfig(backend_id="offline"),
        _inspection(),
        unavailable,
    )
    missing_capability = select_runtime_backend(
        RuntimeConfig(backend_id="available", required_capabilities=("needed.v1",)),
        _inspection(),
        available,
    )

    assert _codes(missing_backend) == ["runtime_backend_not_found"]
    assert _codes(unavailable_backend) == ["runtime_backend_unavailable"]
    assert _codes(missing_capability) == ["runtime_capability_missing"]
    assert missing_capability.missing_capabilities == ("needed.v1",)


def test_platform_request_requires_visible_target_and_fallback_is_explicit() -> None:
    registry = RuntimeBackendRegistry()
    registry.register(
        FakeRuntimeBackend(backend_id="portable", supported_platforms=("cpu", "gpu"))
    )
    inspection = _inspection(platform="cpu")

    blocked = select_runtime_backend(
        RuntimeConfig(backend_id="portable", platform_preference="gpu"),
        inspection,
        registry,
    )
    fallback = select_runtime_backend(
        RuntimeConfig(
            backend_id="portable",
            platform_preference="gpu",
            fallback_policy="allow_compatible",
        ),
        inspection,
        registry,
    )

    assert _codes(blocked) == [
        "requested_platform_unavailable",
        "runtime_fallback_disallowed",
    ]
    assert fallback.status == "pass"
    assert fallback.selected_platform == "cpu"
    assert fallback.fallback_used is True
    assert "runtime_compatible_fallback_used" in _warning_codes(fallback)


def test_automatic_platform_is_documented_and_unspecified_is_not_automatic() -> None:
    registry = RuntimeBackendRegistry()
    registry.register(
        FakeRuntimeBackend(backend_id="portable", supported_platforms=("cpu", "gpu"))
    )
    inspection = _inspection(platform="cpu", additional_platform="gpu")

    automatic = select_runtime_backend(
        RuntimeConfig(platform_preference="automatic"), inspection, registry
    )
    unspecified = select_runtime_backend(RuntimeConfig(), inspection, registry)

    assert automatic.selected_platform == "gpu"
    assert "runtime_platform_inferred" in _warning_codes(automatic)
    assert unspecified.selected_platform is None
    assert "runtime_platform_inferred" not in _warning_codes(unspecified)


def test_selection_rejects_declared_policy_gaps_and_leaves_precision_unevaluated() -> (
    None
):
    registry = RuntimeBackendRegistry()
    registry.register(FakeRuntimeBackend(backend_id="minimal"))

    blocked = select_runtime_backend(
        RuntimeConfig(
            backend_id="minimal",
            platform_preference="cpu",
            placement_policy="replicated",
            compilation_policy="jit",
            distributed_policy="required",
        ),
        _inspection(),
        registry,
    )
    precision = select_runtime_backend(
        RuntimeConfig(
            backend_id="minimal",
            platform_preference="cpu",
            precision_policy="bfloat16",
        ),
        _inspection(),
        registry,
    )

    assert _codes(blocked) == [
        "runtime_policy_unsupported",
        "runtime_policy_unsupported",
        "runtime_policy_unsupported",
    ]
    assert precision.status == "pass"
    assert "runtime_precision_unevaluated" in _warning_codes(precision)


def test_selection_uses_stable_backend_id_tiebreak_without_initialization() -> None:
    registry = RuntimeBackendRegistry()
    registry.register(_NoInitializeBackend("zeta"))
    registry.register(_NoInitializeBackend("alpha"))

    selection = select_runtime_backend(
        RuntimeConfig(platform_preference="cpu"), _inspection(), registry
    )

    assert selection.status == "pass"
    assert selection.selected_backend is not None
    assert selection.selected_backend.backend_id == "alpha"
    assert "runtime_selection_used_tiebreak" in _warning_codes(selection)


class _NoInitializeBackend(FakeRuntimeBackend):
    def initialize(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("selection must not initialize a backend")


def _inspection(
    *,
    jax_available: bool = True,
    platform: str = "cpu",
    additional_platform: str | None = None,
) -> RuntimeInspection:
    platforms = (
        (platform,) if additional_platform is None else (platform, additional_platform)
    )
    devices = tuple(
        DeviceDescriptor(device_id=f"device-{index}", platform=item)
        for index, item in enumerate(platforms)
    )
    return RuntimeInspection(
        status="pass",
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=jax_available,
            platform=platform,
            process_count=1,
            process_index=0,
            local_device_count=len(devices),
            global_device_count=len(devices),
            distributed_initialized=False,
        ),
        device_inventory=DeviceInventory(
            devices=devices,
            process_count=1,
            local_device_count=len(devices),
            global_device_count=len(devices),
        ),
    )


def _codes(selection: RuntimeSelectionResult) -> list[str]:
    return [item.code for item in selection.blockers]


def _warning_codes(selection: RuntimeSelectionResult) -> list[str]:
    return [item.code for item in selection.warnings]
