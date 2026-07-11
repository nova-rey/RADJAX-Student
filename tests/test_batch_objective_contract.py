from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.learning import (
    BatchMetadata,
    LearningBatch,
    MetricRecord,
    ObjectiveRequest,
    ObjectiveResult,
    ObjectiveScope,
    WeightingPolicy,
    canonical_objective_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generic_batch_metadata_and_objective_round_trip() -> None:
    batch = LearningBatch(
        batch_id="batch-1",
        inputs={"tokens": {"count": 2}},
        targets={"labels": {"count": 2}},
        weights={"sample_weights": [1.0, 0.5]},
        objective_scope=ObjectiveScope(kind="intermediate_surface", target_id="hidden"),
    )
    metadata = BatchMetadata(
        sample_count=2,
        sequence_length=4,
        padding_policy="right",
        source="behavior_adapter",
    )
    request = ObjectiveRequest(
        "weighted_loss",
        batch.objective_scope,
        batch.batch_id,
        ("hidden",),
        WeightingPolicy(kind="explicit_weights", weight_key="sample_weights"),
    )
    result = ObjectiveResult(
        "weighted_loss",
        0.25,
        {"main": 0.25},
        (MetricRecord("objective.loss", 0.25, 0),),
    )
    assert LearningBatch.from_dict(batch.to_dict()) == batch
    assert BatchMetadata.from_dict(metadata.to_dict()) == metadata
    assert ObjectiveRequest.from_dict(request.to_dict()) == request
    assert ObjectiveResult.from_dict(result.to_dict()) == result
    assert json.loads(canonical_objective_json(result.to_dict()))["loss"] == 0.25


def test_weighting_policy_and_objective_scope_validation() -> None:
    assert WeightingPolicy().kind == "uniform"
    with pytest.raises(ValueError, match="requires weight_key"):
        WeightingPolicy(kind="explicit_weights")
    with pytest.raises(ValueError, match="uniform weighting"):
        WeightingPolicy(weight_key="weights")
    request = ObjectiveRequest("loss", ObjectiveScope(), required_outputs=("logits",))
    assert request.objective_scope.kind == "final_output"


def test_batch_objective_contract_import_isolation() -> None:
    script = """
import builtins
import sys
real_import = builtins.__import__
forbidden = {"jax", "jaxlib", "torch", "transformers", "datasets", "radjax_tome"}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from radjax_student.learning import LearningBatch, ObjectiveRequest
assert LearningBatch(batch_id="b").objective_scope.kind == "final_output"
assert ObjectiveRequest("o").objective_id == "o"
assert not any(name.startswith("radjax_student.architecture") for name in sys.modules)
assert not any(name.startswith("radjax_student.runtime") for name in sys.modules)
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
