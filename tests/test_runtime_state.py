from __future__ import annotations

import json
import socket
import urllib.request
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from radjax_student.runtime import (
    RUNTIME_STATE_FILE,
    RUNTIME_STATE_MANIFEST_FILE,
    RuntimeConfig,
    RuntimeContractError,
    RuntimeState,
    canonical_runtime_state_json,
    evaluate_runtime_resume_compatibility,
    load_runtime_state,
    load_runtime_state_with_receipt,
    save_runtime_state,
)
from radjax_student.runtime import smoke as smoke_module


def test_runtime_state_round_trip_is_deterministic_runtime_only(tmp_path: Path) -> None:
    state = _state()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_receipt = save_runtime_state(state, first)
    second_receipt = save_runtime_state(state, second)
    loaded, load_receipt = load_runtime_state_with_receipt(first)

    assert loaded == state
    assert first_receipt.hashes == second_receipt.hashes
    assert (first / RUNTIME_STATE_FILE).read_bytes() == (
        second / RUNTIME_STATE_FILE
    ).read_bytes()
    assert load_receipt.verified_files == (
        RUNTIME_STATE_FILE,
        RUNTIME_STATE_MANIFEST_FILE,
    )
    payload = json.loads((first / RUNTIME_STATE_FILE).read_text(encoding="utf-8"))
    assert payload["runtime_keys"]["root_seed"] == 19
    assert "model_parameters" not in payload
    assert "optimizer_state" not in payload
    assert "compiled_executables" not in payload
    assert "jax_devices" not in payload


def test_runtime_state_refuses_implicit_overwrite_and_accepts_explicit_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "runtime_state"
    save_runtime_state(_state(), destination)

    with pytest.raises(RuntimeContractError) as error:
        save_runtime_state(_state(global_step=2), destination)
    assert error.value.code == "runtime_state_exists"

    receipt = save_runtime_state(_state(global_step=2), destination, overwrite=True)
    assert receipt.status == "pass"
    assert load_runtime_state(destination).global_step == 2


def test_runtime_state_rejects_hash_size_missing_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "runtime_state"
    save_runtime_state(_state(), destination)
    state_path = destination / RUNTIME_STATE_FILE

    state_path.write_bytes(state_path.read_bytes() + b" ")
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_size_mismatch"

    state_path.unlink()
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_missing"

    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(tmp_path / "missing")
    assert error.value.code == "runtime_state_missing"

    with pytest.raises(RuntimeContractError) as error:
        save_runtime_state(_state(), tmp_path / "safe" / ".." / "unsafe")
    assert error.value.code == "runtime_state_path_unsafe"


def test_runtime_state_rejects_malformed_and_unsupported_payloads(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "runtime_state"
    save_runtime_state(_state(), destination)
    (destination / RUNTIME_STATE_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_size_mismatch"

    save_runtime_state(_state(), destination, overwrite=True)
    _rewrite_state(
        destination,
        lambda payload: payload.__setitem__("schema_version", "runtime_state.v2"),
    )
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_schema_unsupported"

    save_runtime_state(_state(), destination, overwrite=True)
    manifest_path = destination / RUNTIME_STATE_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "runtime_state.v2"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_schema_unsupported"


def test_runtime_state_revalidates_rng_and_rejects_forbidden_state(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "runtime_state"
    save_runtime_state(_state(), destination)

    def tamper_rng(payload: dict) -> None:
        payload["runtime_keys"]["streams"][0]["derived_seed"] += 1

    _rewrite_state(destination, tamper_rng)
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_rng_invalid"

    save_runtime_state(_state(), destination, overwrite=True)
    _rewrite_state(
        destination,
        lambda payload: payload["runtime_keys"].__setitem__("root_seed", 20),
    )
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_rng_invalid"

    save_runtime_state(_state(), destination, overwrite=True)
    _rewrite_state(
        destination, lambda payload: payload.__setitem__("model_parameters", {})
    )
    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_manifest_invalid"


def test_runtime_state_rejects_unsafe_manifest_internal_file_name(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "runtime_state"
    save_runtime_state(_state(), destination)
    manifest_path = destination / RUNTIME_STATE_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = ["../runtime_state.json"]
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(RuntimeContractError) as error:
        load_runtime_state(destination)
    assert error.value.code == "runtime_state_manifest_invalid"


def test_resume_compatibility_reports_runtime_policy_not_architecture() -> None:
    state = _state()

    passing = evaluate_runtime_resume_compatibility(state, state.runtime_config)
    failing = evaluate_runtime_resume_compatibility(
        state,
        RuntimeConfig(
            backend_id="other",
            precision_policy="bfloat16",
            placement_policy="replicated",
            distributed_policy="required",
            seed=20,
        ),
    )

    assert passing.ok
    assert "execution_equivalence" in passing.claims_not_made[0]
    assert not failing.ok
    assert {
        item.details["field"] for item in failing.blockers if "field" in item.details
    } == {
        "precision_policy",
        "placement_policy",
        "distributed_policy",
    }


def test_state_save_and_load_do_not_use_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    destination = tmp_path / "runtime_state"
    save_runtime_state(_state(), destination)
    assert load_runtime_state(destination).runtime_id == "runtime-state-test"


def test_runtime_state_smoke_and_doctor_option_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radjax_student.cli.main import main
    from radjax_student.reports import doctor as doctor_module
    from radjax_student.runtime import state as state_module

    environment = SimpleNamespace(
        platform="cpu",
        process_count=1,
        process_index=0,
        to_dict=lambda: {"platform": "cpu", "process_count": 1},
    )
    inventory = SimpleNamespace(
        devices=(),
        local_device_count=1,
        global_device_count=1,
        topology_summary={},
    )
    runtime_report = SimpleNamespace(
        environment=environment,
        device_inventory=inventory,
    )
    receipt = SimpleNamespace(
        ok=True,
        runtime_id="runtime-state-smoke",
        backend_id="fake",
        config=_state().runtime_config,
        runtime_report=runtime_report,
        blockers=(),
    )
    monkeypatch.setattr(
        smoke_module,
        "run_single_device_cpu_smoke",
        lambda *args: receipt,
    )

    smoke_receipt = state_module.run_runtime_state_smoke()
    assert smoke_receipt.ok
    assert smoke_receipt.global_step == 3
    assert smoke_receipt.model_state_included is False
    assert smoke_receipt.optimizer_state_included is False

    monkeypatch.setattr(doctor_module, "run_runtime_state_smoke", lambda: smoke_receipt)
    stdout = StringIO()
    code = main(("doctor", "--runtime-state-smoke", "--format", "json"), stdout=stdout)
    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["runtime_state_smoke"]["status"] == "pass"


def _state(*, global_step: int = 1) -> RuntimeState:
    config = RuntimeConfig(
        backend_id="fake",
        platform_preference="cpu",
        precision_policy="float32",
        placement_policy="single_device",
        compilation_policy="eager",
        distributed_policy="disabled",
        seed=19,
    )
    return RuntimeState(
        runtime_id="runtime-state-test",
        global_step=global_step,
        root_seed=19,
        runtime_config=config,
        environment_summary={"platform": "cpu", "process_count": 1},
        topology_summary={
            "platform": "cpu",
            "process_count": 1,
            "global_device_count": 1,
            "device_kinds": ["fake-cpu"],
        },
        precision_policy="float32",
        placement_policy="single_device",
        backend_id="fake",
        resume_metadata={"cursor": "runtime-only"},
    )


def _rewrite_state(destination: Path, mutate) -> None:
    state_path = destination / RUNTIME_STATE_FILE
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    mutate(payload)
    state_bytes = canonical_runtime_state_json(payload)
    state_path.write_bytes(state_bytes)
    manifest_path = destination / RUNTIME_STATE_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hashes"][RUNTIME_STATE_FILE] = _sha256(state_bytes)
    manifest["sizes"][RUNTIME_STATE_FILE] = len(state_bytes)
    manifest["integrity"]["state_digest"] = _sha256(state_bytes)
    _rewrite_manifest(manifest_path, manifest)


def _rewrite_manifest(path: Path, manifest: dict) -> None:
    base = dict(manifest)
    base.pop("integrity", None)
    manifest["integrity"]["manifest_digest"] = _sha256(
        canonical_runtime_state_json(base)
    )
    path.write_bytes(canonical_runtime_state_json(manifest))


def _sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()
