from __future__ import annotations

import json

import pytest

from radjax_student.architecture import ArchitectureState
from radjax_student.checkpoints import (
    LearningCheckpoint,
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from radjax_student.learning import LearningState
from radjax_student.optimizers import OptimizerState


def _checkpoint() -> LearningCheckpoint:
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
        {},
        {},
    )


def test_layered_checkpoint_round_trip_and_manifest_is_deterministic(tmp_path):
    saved = save_learning_checkpoint(_checkpoint(), tmp_path)
    restored = load_learning_checkpoint(tmp_path, runtime_reference="runtime-1")
    assert restored.parameters == {"head.weight": 2.0}
    assert restored.optimizer_state.backend_state == {
        "per_parameter_steps": {"head.weight": 1}
    }
    assert saved.integrity == load_learning_checkpoint(tmp_path).integrity


def test_checkpoint_rejects_hash_schema_path_and_runtime_mismatches(tmp_path):
    save_learning_checkpoint(_checkpoint(), tmp_path)
    with pytest.raises(ValueError, match="runtime reference"):
        load_learning_checkpoint(tmp_path, runtime_reference="other")
    (tmp_path / "architecture.json").write_text("{}")
    with pytest.raises(ValueError, match="component hash"):
        load_learning_checkpoint(tmp_path)
    save_learning_checkpoint(_checkpoint(), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["schema_version"] = "wrong"
    manifest["integrity"]["manifest_digest"] = (
        __import__("hashlib")
        .sha256(
            (
                json.dumps(
                    {
                        key: value
                        for key, value in manifest.items()
                        if key != "integrity"
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
        .hexdigest()
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema"):
        load_learning_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="paths"):
        LearningCheckpoint(
            "r",
            LearningState(run_id="r", optimizer_step=0),
            None,
            OptimizerState("sgd.v1", ("a",)),
            {"b": 1.0},
            {},
            {},
        )
