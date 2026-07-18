from pathlib import Path

import pytest
from radjax_contract.io import read_json, write_json

from radjax_student.artifacts import TomeArtifactError, open_tome_artifact
from tests.support.tome_fixtures import write_dense_tome


def test_open_tome_artifact_returns_stable_metadata_view(tmp_path: Path) -> None:
    tome = write_dense_tome(tmp_path / "tome")

    view = open_tome_artifact(tome)

    assert view.artifact_dir == tome
    assert view.manifest.producer == "radjax-tome"
    assert view.cover_page.title == "Tiny dense logits smoke Tome"
    assert view.payload_format.value == "dense_logits_v0"
    assert view.sequence_length == 2
    assert view.record_count == 2
    assert view.vocab_contract is not None
    assert view.vocab_contract.tokenizer_id == "toy"
    assert view.tokenizer_contract == {
        "tokenizer_id": "toy",
        "tokenizer_hash": None,
        "model_id": None,
        "model_family": None,
        "special_tokens": {},
    }
    assert view.payload_summary.expected_adapter == "dense_logits"
    assert view.payload_summary.implemented_by_contract is True
    assert view.payload_summary.shard_paths == ()
    assert view.inferred_defaults.teacher_family == "fake"
    assert view.inferred_defaults.adapter_family == "dense_logits"
    assert view.inferred_defaults.requires_reconstruction is False


def test_open_tome_artifact_reports_missing_cover_page(tmp_path: Path) -> None:
    tome = write_dense_tome(tmp_path / "tome")
    (tome / "cover_page.json").unlink()

    with pytest.raises(TomeArtifactError) as exc_info:
        open_tome_artifact(tome)

    assert "cover_page_missing" in exc_info.value.blockers


def test_open_tome_artifact_reports_cover_manifest_disagreement(
    tmp_path: Path,
) -> None:
    tome = write_dense_tome(tmp_path / "tome")
    cover = read_json(tome / "cover_page.json")
    cover["contents"]["record_count"] = 999
    write_json(tome / "cover_page.json", cover)

    with pytest.raises(TomeArtifactError) as exc_info:
        open_tome_artifact(tome)

    assert any(
        "cover_record_count_mismatch" in item for item in exc_info.value.blockers
    )


def test_open_tome_artifact_reports_malformed_manifest(tmp_path: Path) -> None:
    tome = write_dense_tome(tmp_path / "tome")
    (tome / "manifest.json").write_text("{ nope", encoding="utf-8")

    with pytest.raises(TomeArtifactError) as exc_info:
        open_tome_artifact(tome)

    assert any("manifest_malformed_json" in item for item in exc_info.value.blockers)
