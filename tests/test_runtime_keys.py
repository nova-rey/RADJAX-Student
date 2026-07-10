from __future__ import annotations

import json

import pytest

from radjax_student.runtime import (
    RUNTIME_KEY_STREAM_NAMES,
    DeviceInventory,
    ExecutionContext,
    RuntimeCapabilityProfile,
    RuntimeEnvironment,
    RuntimeKeys,
)


def test_identical_seed_reproduces_the_complete_named_tree() -> None:
    first = RuntimeKeys.from_seed(1234)
    second = RuntimeKeys.from_seed(1234)

    assert first == second
    assert tuple(item.name for item in first.streams) == RUNTIME_KEY_STREAM_NAMES
    assert first.model_initialization == first.stream("model_initialization")
    assert first.runtime_tests == first.stream("runtime_tests")


def test_different_roots_and_named_streams_are_isolated() -> None:
    keys = RuntimeKeys.from_seed(1234)
    other = RuntimeKeys.from_seed(1235)
    derived = tuple(item.derived_seed for item in keys.streams)

    assert keys != other
    assert len(set(derived)) == len(RUNTIME_KEY_STREAM_NAMES)
    assert keys.dropout.derived_seed != keys.augmentation.derived_seed
    assert keys.data_order.lineage == ("root_seed", "data_order")


def test_runtime_keys_round_trip_is_deterministic_json_metadata_only() -> None:
    keys = RuntimeKeys.from_seed(7)
    payload = keys.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert RuntimeKeys.from_dict(payload) == keys
    assert json.loads(encoded) == payload
    assert "jax" not in encoded
    assert "numpy" not in encoded


def test_execution_context_derives_and_serializes_runtime_keys_from_root_seed() -> None:
    context = _context()
    payload = context.to_dict()

    assert context.runtime_keys == RuntimeKeys.from_seed(7)
    assert payload["runtime_keys"]["root_seed"] == 7
    assert ExecutionContext.from_dict(payload) == context


def test_execution_context_rejects_runtime_key_root_mismatch() -> None:
    with pytest.raises(ValueError, match="root seed"):
        _context(runtime_keys=RuntimeKeys.from_seed(8))


def test_runtime_keys_reject_unknown_or_reordered_stream_contract() -> None:
    keys = RuntimeKeys.from_seed(1)
    payload = keys.to_dict()
    payload["streams"].reverse()

    with pytest.raises(ValueError, match="contract order"):
        RuntimeKeys.from_dict(payload)
    with pytest.raises(KeyError, match="unknown runtime key stream"):
        keys.stream("unknown")
    with pytest.raises(ValueError, match="nonnegative"):
        RuntimeKeys.from_seed(-1)


def test_runtime_keys_module_has_no_backend_or_global_rng_dependencies() -> None:
    source = (
        _repo_root() / "src" / "radjax_student" / "runtime" / "keys.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "import random",
        "import numpy",
        "import jax",
        "radjax_student.students",
        "radjax_student.training",
        "radjax_student.artifacts",
    ):
        assert forbidden not in source


def _context(*, runtime_keys: RuntimeKeys | None = None) -> ExecutionContext:
    return ExecutionContext(
        backend_id="fake",
        environment=RuntimeEnvironment(
            python_version="3.11.9",
            jax_available=False,
            process_count=1,
            process_index=0,
            local_device_count=1,
            global_device_count=1,
            distributed_initialized=False,
        ),
        device_inventory=DeviceInventory(
            process_count=1,
            local_device_count=1,
            global_device_count=1,
        ),
        capabilities=RuntimeCapabilityProfile(
            profile_id="fake.runtime.v1",
            backend_id="fake",
            version=1,
            capabilities=("runtime.single_process_v1",),
        ),
        root_seed=7,
        runtime_id="runtime-test-1",
        runtime_keys=runtime_keys,
    )


def _repo_root():
    return __import__("pathlib").Path(__file__).resolve().parents[1]
