from __future__ import annotations

import json

import pytest

from radjax_student.architecture import ArchitectureState
from radjax_student.checkpoints import (
    CONTINUATION_CHECKPOINT_ROLE,
    HF_DISTRIBUTION_CHECKPOINT_ROLE,
    CheckpointPayloadDescriptor,
    FutureTensorPayloadDescriptor,
    LearningCheckpoint,
    load_learning_checkpoint,
    reject_implicit_hf_conversion,
    save_learning_checkpoint,
)
from radjax_student.learning import LearningState
from radjax_student.optimizers import OptimizerState


def _checkpoint(source_state=None):
    return LearningCheckpoint(
        "runtime-reference",
        LearningState("p35", global_step=1, optimizer_step=1),
        ArchitectureState("p35.architecture"),
        OptimizerState("sgd.v1", ("head.weight",), step=1, backend_state={"step": 1}),
        {"head.weight": 0.5},
        source_state,
        {},
        {},
    )


def test_continuation_checkpoint_declares_continuation_role(tmp_path):
    saved = save_learning_checkpoint(_checkpoint({"position": 1}), tmp_path)
    assert saved.role == CONTINUATION_CHECKPOINT_ROLE


def test_continuation_checkpoint_rejects_hf_distribution_role():
    with pytest.raises(ValueError, match="HF distribution"):
        LearningCheckpoint(
            "runtime",
            LearningState("p35", global_step=1, optimizer_step=1),
            None,
            OptimizerState("sgd.v1", ("head.weight",), step=1),
            {"head.weight": 1.0},
            None,
            {},
            {},
            role=HF_DISTRIBUTION_CHECKPOINT_ROLE,
        )


def test_implicit_hf_conversion_is_rejected():
    with pytest.raises(ValueError, match="cannot be treated"):
        reject_implicit_hf_conversion(_checkpoint())


def test_v2_architecture_payload_is_truthful_scalar_mapping(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "learning_checkpoint.v2"
    assert (
        manifest["payload_descriptors"]["architecture.json"]["kind"]
        == "scalar_parameter_mapping"
    )


def test_v2_manifest_does_not_emit_tensor_payload_descriptor(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert all(
        "tensor" not in value["kind"]
        for value in manifest["payload_descriptors"].values()
    )


def test_payload_descriptor_round_trip_is_typed_and_deterministic():
    descriptor = CheckpointPayloadDescriptor(
        "architecture", "json", "scalar_parameter_mapping"
    )
    assert CheckpointPayloadDescriptor.from_dict(descriptor.to_dict()) == descriptor


def test_payload_descriptor_rejects_empty_owner():
    with pytest.raises(ValueError, match="nonempty"):
        CheckpointPayloadDescriptor("", "json", "state")


def test_payload_descriptor_rejects_unknown_schema():
    with pytest.raises(ValueError, match="unsupported"):
        CheckpointPayloadDescriptor("learning", "json", "state", "other")


def test_future_tensor_descriptor_is_declarative_only():
    descriptor = FutureTensorPayloadDescriptor("jax_pytree.v1", "future_codec")
    assert descriptor.to_dict()["storage_codec"] == "future_codec"


def test_checkpoint_manifest_rejects_invalid_descriptor(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_descriptors"]["architecture.json"]["kind"] = "pytree_reference"
    manifest.pop("integrity")
    import hashlib

    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    manifest["integrity"] = {
        "algorithm": "sha256",
        "manifest_digest": hashlib.sha256(payload.encode()).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(ValueError, match="payload descriptors"):
        load_learning_checkpoint(tmp_path)


def test_checkpoint_source_none_round_trips(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    assert load_learning_checkpoint(tmp_path).source_state is None


def test_checkpoint_rejects_runtime_reference_mismatch(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    with pytest.raises(ValueError, match="runtime reference"):
        load_learning_checkpoint(tmp_path, runtime_reference="other")
