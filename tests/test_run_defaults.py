from __future__ import annotations

import json
from pathlib import Path

import pytest
from tome_fixtures import write_dense_tome

from radjax_student.artifacts import TomeArtifactError, open_tome_artifact
from radjax_student.validation import (
    infer_run_defaults,
    infer_run_defaults_from_tome,
)


def test_infer_run_defaults_separates_inferred_user_and_phase_values(
    tmp_path: Path,
) -> None:
    view = open_tome_artifact(write_dense_tome(tmp_path / "tome"))

    defaults = infer_run_defaults(view)

    assert defaults.inferred_from_tome == {
        "teacher_id": "fake-teacher",
        "teacher_family": "fake",
        "teacher_backend": "fake",
        "tokenizer_id": "toy",
        "vocab_size": 3,
        "sequence_length": 2,
        "record_count": 2,
        "payload_format": "dense_logits_v0",
        "compression_family": "none",
        "requires_reconstruction": False,
        "expected_adapter_family": "dense_logits",
        "artifact_role": "training",
    }
    assert defaults.required_from_user == {
        "student_architecture": None,
        "student_size_or_config": None,
        "training_budget": None,
        "output_dir": None,
    }
    assert defaults.unresolved_by_phase == {
        "runtime_backend": "phase_2_or_later",
        "optimizer": "phase_3_or_later",
        "schedule_policy": "phase_3_or_later",
        "hf_export_details": "phase_5_or_later",
        "evaluation_policy": "phase_5_or_later",
    }


def test_run_defaults_report_claims_not_made_and_serializes(
    tmp_path: Path,
) -> None:
    defaults = infer_run_defaults_from_tome(write_dense_tome(tmp_path / "tome"))

    payload = defaults.to_dict()

    assert "training_not_run" in payload["claims_not_made"]
    assert "student_architecture_not_selected" in payload["claims_not_made"]
    assert "runtime_not_selected" in payload["claims_not_made"]
    assert "compatibility_not_passed" in payload["claims_not_made"]
    assert "model_quality_not_claimed" in payload["claims_not_made"]
    assert "hf_export_not_ready" in payload["claims_not_made"]
    assert "radlads_parity_not_measured" in payload["claims_not_made"]
    json.dumps(payload)


def test_run_defaults_path_helper_keeps_malformed_tome_failures_in_artifacts(
    tmp_path: Path,
) -> None:
    tome = write_dense_tome(tmp_path / "tome")
    (tome / "cover_page.json").unlink()

    with pytest.raises(TomeArtifactError):
        infer_run_defaults_from_tome(tome)
