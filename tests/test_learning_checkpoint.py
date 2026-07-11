from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from radjax_student.architecture import ArchitectureState
from radjax_student.checkpoints import (
    CONTINUATION_CHECKPOINT_ROLE,
    LearningCheckpoint,
    load_learning_checkpoint,
    reject_implicit_hf_conversion,
    save_learning_checkpoint,
)
from radjax_student.learning import LearningState
from radjax_student.optimizers import OptimizerState


def _checkpoint(source_state=None) -> LearningCheckpoint:
    state = OptimizerState(
        "sgd.v1",
        ("head.weight",),
        step=1,
        backend_state={"per_parameter_steps": {"head.weight": 1}},
    )
    return LearningCheckpoint(
        "runtime-1",
        LearningState(run_id="run", global_step=1, optimizer_step=1),
        ArchitectureState("architecture-1"),
        state,
        {"head.weight": 2.0},
        {"source_id": "fixture", "position": 1}
        if source_state is None
        else source_state,
        {},
        {},
    )


def _rewrite_manifest(path, mutate) -> None:
    manifest = json.loads(path.read_text())
    mutate(manifest)
    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    manifest["integrity"] = {
        "algorithm": "sha256",
        "manifest_digest": hashlib.sha256(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
    }
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


def test_layered_checkpoint_round_trip_and_manifest_is_deterministic(tmp_path):
    saved = save_learning_checkpoint(_checkpoint(), tmp_path)
    restored = load_learning_checkpoint(tmp_path, runtime_reference="runtime-1")
    assert restored.parameters == {"head.weight": 2.0}
    assert restored.optimizer_state.backend_state == {
        "per_parameter_steps": {"head.weight": 1}
    }
    assert restored.source_state == {"position": 1, "source_id": "fixture"}
    assert saved.integrity == load_learning_checkpoint(tmp_path).integrity


def test_checkpoint_rejects_runtime_hash_and_path_mismatches(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    with pytest.raises(ValueError, match="runtime reference"):
        load_learning_checkpoint(tmp_path, runtime_reference="other")
    (tmp_path / "architecture.json").write_text("{}")
    with pytest.raises(ValueError, match="component hash"):
        load_learning_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="paths"):
        LearningCheckpoint(
            "r",
            LearningState(run_id="r", optimizer_step=0),
            None,
            OptimizerState("sgd.v1", ("a",)),
            {"b": 1.0},
            None,
            {},
            {},
        )


def test_v2_checkpoint_includes_source_component_and_manifest_ownership(tmp_path):
    saved = save_learning_checkpoint(_checkpoint(), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert saved.schema_version == "learning_checkpoint.v2"
    assert (tmp_path / "source.json").is_file()
    assert manifest["files"] == [
        "architecture.json",
        "learning.json",
        "optimizer.json",
        "source.json",
    ]
    assert manifest["ownership"]["source.json"] == "batch_source"
    assert saved.role == CONTINUATION_CHECKPOINT_ROLE
    assert manifest["checkpoint_role"] == CONTINUATION_CHECKPOINT_ROLE
    assert manifest["payload_descriptors"]["architecture.json"]["kind"] == (
        "pytree_reference"
    )


def test_continuation_checkpoint_cannot_be_used_as_hf_distribution(tmp_path):
    checkpoint = _checkpoint()
    save_learning_checkpoint(checkpoint, tmp_path)
    with pytest.raises(ValueError, match="explicit HF distribution conversion"):
        reject_implicit_hf_conversion(checkpoint)


def test_source_component_has_hash_size_and_deterministic_payload(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path / "one")
    save_learning_checkpoint(_checkpoint(), tmp_path / "two")
    one = json.loads((tmp_path / "one" / "manifest.json").read_text())
    two = json.loads((tmp_path / "two" / "manifest.json").read_text())
    assert one["hashes"]["source.json"] == two["hashes"]["source.json"]
    assert one["sizes"]["source.json"] == len(
        (tmp_path / "one" / "source.json").read_bytes()
    )
    assert (tmp_path / "one" / "source.json").read_bytes() == (
        tmp_path / "two" / "source.json"
    ).read_bytes()
    assert one["integrity"]["manifest_digest"] == two["integrity"]["manifest_digest"]


def test_none_source_state_round_trips(tmp_path):
    checkpoint = replace(_checkpoint(), source_state=None)
    save_learning_checkpoint(checkpoint, tmp_path)
    assert load_learning_checkpoint(tmp_path).source_state is None


def test_corrupted_source_payload_is_rejected(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    (tmp_path / "source.json").write_text("{}")
    with pytest.raises(ValueError, match="component hash"):
        load_learning_checkpoint(tmp_path)


def test_missing_source_component_is_rejected(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    (tmp_path / "source.json").unlink()
    with pytest.raises(ValueError, match="component missing"):
        load_learning_checkpoint(tmp_path)


def test_source_hash_mismatch_is_rejected(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    _rewrite_manifest(
        tmp_path / "manifest.json",
        lambda manifest: manifest["hashes"].update({"source.json": "wrong"}),
    )
    with pytest.raises(ValueError, match="component hash"):
        load_learning_checkpoint(tmp_path)


def test_source_size_mismatch_is_rejected(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    _rewrite_manifest(
        tmp_path / "manifest.json",
        lambda manifest: manifest["sizes"].update({"source.json": 0}),
    )
    with pytest.raises(ValueError, match="size mismatch"):
        load_learning_checkpoint(tmp_path)


def test_source_manifest_omission_is_rejected(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)

    def omit_source(manifest):
        manifest["files"].remove("source.json")
        manifest["hashes"].pop("source.json")
        manifest["sizes"].pop("source.json")
        manifest["ownership"].pop("source.json")

    _rewrite_manifest(tmp_path / "manifest.json", omit_source)
    with pytest.raises(ValueError, match="components"):
        load_learning_checkpoint(tmp_path)


def test_source_ownership_mismatch_is_rejected(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    _rewrite_manifest(
        tmp_path / "manifest.json",
        lambda manifest: manifest["ownership"].update({"source.json": "learning"}),
    )
    with pytest.raises(ValueError, match="ownership"):
        load_learning_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "source_state, error",
    [
        ({"bad": {1, 2}}, TypeError),
        ({"bad": b"bytes"}, TypeError),
        ({"bad": lambda: None}, TypeError),
        ({1: "bad"}, TypeError),
        ({"bad": float("nan")}, ValueError),
        ({"bad": float("inf")}, ValueError),
    ],
)
def test_invalid_source_state_is_rejected(source_state, error):
    with pytest.raises(error):
        replace(_checkpoint(), source_state=source_state)


def test_unsupported_checkpoint_schema_is_rejected(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    _rewrite_manifest(
        tmp_path / "manifest.json",
        lambda manifest: manifest.update({"schema_version": "learning_checkpoint.v1"}),
    )
    with pytest.raises(ValueError, match="schema"):
        load_learning_checkpoint(tmp_path)
