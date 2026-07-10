from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from radjax_contract.testing import production_tome_fixture_path

from radjax_student.artifacts import (
    TomeArtifactError,
    open_tome_artifact,
)

CURRENT_REQUIRED_CAPABILITIES = {
    "radjax.corridor.packed_assignments.v1",
    "radjax.corridor.stat_bands.v1",
    "radjax.exemplar.selected_dynamic_topk.v1",
}


def test_open_exact_contract_production_fixture() -> None:
    fixture = production_tome_fixture_path()

    view = open_tome_artifact(fixture)

    # The Contract helper owns the accepted P1.5 bytes and digest.
    assert view.artifact_dir == fixture
    assert view.contract_family == "production_v2"
    assert view.identity.artifact_kind == "radjax_tome"
    assert view.identity.cover_page_version == 2
    assert view.identity.tome_version == 1
    assert view.identity.layout == "unpacked_directory"
    assert view.identity.source_artifact_type == "teacher_textbook"
    assert view.provenance.teacher["model_id"] == "fake-production-teacher"
    assert view.provenance.tokenizer["vocab_size"] == 32
    assert view.provenance.corpus is not None
    assert view.provenance.corpus["corpus_hash"] == "sha256:fixture-corpus-v1"
    assert view.validation.producer_status == "pass"
    assert view.validation.contract_status == "pass"
    assert view.validation.blockers == ()
    assert view.validation.student_interpretation == (
        "metadata_only_requires_capabilities"
    )
    assert set(view.validation.required_capabilities) == CURRENT_REQUIRED_CAPABILITIES
    assert set(view.validation.unsupported_required_capabilities) == (
        CURRENT_REQUIRED_CAPABILITIES
    )
    assert "no_student_training_claim" in view.claims_not_made
    assert len(view.contents_index) == 21
    assert all(
        ref.path and ref.sha256 and ref.size_bytes > 0 for ref in view.contents_index
    )
    assert view.manifest is None
    assert view.payload_summary is None
    assert view.payload_format is None


def test_production_surface_collection_and_pass_plan_are_preserved() -> None:
    view = open_tome_artifact(production_tome_fixture_path())

    assert [
        (surface.surface_id, surface.surface_kind) for surface in view.surfaces
    ] == [
        ("corridor", "fingerprint_corridor"),
        ("exemplar", "selected_exemplar"),
    ]
    assert view.surface("corridor") is view.surfaces[0]
    assert view.surface("missing") is None
    assert all(surface.known_surface for surface in view.surfaces)
    assert view.surfaces[0].surface_id != view.surfaces[1].surface_id
    assert view.recommended_training_plan is not None
    passes = view.recommended_training_plan.passes
    assert [training_pass.surface_id for training_pass in passes] == [
        "corridor",
        "exemplar",
    ]
    assert [training_pass.checkpoint_after for training_pass in passes] == [True, True]
    assert passes[1].prerequisites == ("corridor",)


def test_production_convenience_projections_are_metadata_only() -> None:
    view = open_tome_artifact(production_tome_fixture_path())

    corridor = view.corridor_contract
    exemplar = view.exemplar_contract
    assert corridor is not None
    assert corridor.mode_policy == "stat_bands_v0"
    assert corridor.mode_count == 2
    assert corridor.assignment_count == 32
    assert corridor.assignment_storage_kind == "packed_numpy_v1"
    assert corridor.mode_identifier_type is not corridor.fingerprint_identifier_type
    assert not hasattr(corridor, "assignments")
    assert not hasattr(corridor, "fingerprints")
    assert exemplar is not None
    assert exemplar.selected_exemplar_count == 4
    assert exemplar.dynamic_top_k_metadata == {
        "dynamic_mass_threshold": "0.95",
        "dynamic_top_k_max": "5",
        "dynamic_top_k_min": "2",
    }
    assert exemplar.corridor_linkage_required
    assert len(exemplar.payload_shard_references) == 1
    assert not hasattr(exemplar, "payloads")
    assert not hasattr(view, "model")


def test_normalized_production_metadata_is_immutable() -> None:
    view = open_tome_artifact(production_tome_fixture_path())

    with pytest.raises(TypeError):
        view.provenance.teacher["model_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        view.surfaces[0].semantics["mode_policy"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("path_traversal", "content_path_unsafe"),
        ("stale_hash", "content_hash_mismatch"),
        ("stale_size", "content_size_mismatch"),
        ("missing_required_role", "surface_required_role_missing"),
        ("missing_pass_surface", "training_pass_surface_missing"),
        ("mode_fingerprint_confusion", "corridor_mode_id_domain_invalid"),
    ],
)
def test_contract_blockers_survive_student_error_normalization(
    tmp_path: Path,
    mutation: str,
    blocker: str,
) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    if mutation == "path_traversal":
        _content_ref(cover, "corridor_summary")["path"] = "../outside.json"
    elif mutation == "stale_hash":
        _content_ref(cover, "corridor_summary")["sha256"] = "0" * 64
    elif mutation == "stale_size":
        _content_ref(cover, "corridor_summary")["size_bytes"] += 1
    elif mutation == "missing_required_role":
        cover["contents"] = [
            item for item in cover["contents"] if item["role"] != "corridor_summary"
        ]
    elif mutation == "missing_pass_surface":
        cover["recommended_training_plan"]["passes"][0]["surface_id"] = "missing"
    elif mutation == "mode_fingerprint_confusion":
        mode_ref = _content_ref(cover, "corridor_assignment_mode_id")
        fingerprint_ref = _content_ref(
            cover,
            "corridor_assignment_fingerprint_index",
        )
        mode_ref.update(
            path=fingerprint_ref["path"],
            sha256=fingerprint_ref["sha256"],
            size_bytes=fingerprint_ref["size_bytes"],
        )
    _write_json(artifact / "cover_page.json", cover)

    with pytest.raises(TomeArtifactError) as exc_info:
        open_tome_artifact(artifact)

    assert any(blocker in item for item in exc_info.value.blockers)
    assert blocker in str(exc_info.value)


def test_unknown_optional_role_and_surface_remain_inspectable(
    tmp_path: Path,
) -> None:
    artifact = _copy_fixture(tmp_path)
    future_path = artifact / "future-optional.json"
    _write_json(future_path, {"future": True})
    cover = _read_json(artifact / "cover_page.json")
    cover["contents"].append(
        {
            "classification": "diagnostic",
            "path": "future-optional.json",
            "required": False,
            "role": "future_optional_diagnostic",
            "sha256": hashlib.sha256(future_path.read_bytes()).hexdigest(),
            "size_bytes": future_path.stat().st_size,
        }
    )
    cover["behavioral_surfaces"].append(
        {
            "optional_content_roles": ["future_optional_diagnostic"],
            "prerequisites": [],
            "required_capabilities": [],
            "required_content_roles": [],
            "schema_version": "future_surface_v1",
            "semantics": {"future_field": True},
            "surface_id": "future_optional",
            "surface_kind": "future_optional_kind",
            "target_scope": {"kind": "plugin_defined", "plugin": "future"},
        }
    )
    _write_json(artifact / "cover_page.json", cover)

    view = open_tome_artifact(artifact)

    future_ref = next(
        ref for ref in view.contents_index if ref.role == "future_optional_diagnostic"
    )
    assert not future_ref.known_role
    future_surface = view.surface("future_optional")
    assert future_surface is not None
    assert not future_surface.known_surface
    assert future_surface.target_scope["kind"] == "plugin_defined"
    assert "unknown_content_role: future_optional_diagnostic" in view.warnings
    assert any("unknown_surface_kind" in warning for warning in view.warnings)


def test_unknown_required_capability_is_explicit_without_parse_failure(
    tmp_path: Path,
) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    future_capability = "radjax.future.required.v1"
    cover["behavioral_surfaces"][0]["required_capabilities"].append(future_capability)
    cover["recommended_training_plan"]["passes"][0]["required_capabilities"].append(
        future_capability
    )
    _write_json(artifact / "cover_page.json", cover)

    view = open_tome_artifact(artifact)

    assert view.validation.contract_status == "pass"
    assert future_capability in view.validation.unsupported_required_capabilities
    assert f"unsupported_required_capability: {future_capability}" in view.warnings


def test_malformed_production_cover_does_not_fall_back_silently(
    tmp_path: Path,
) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    del cover["behavioral_surfaces"]
    _write_json(artifact / "cover_page.json", cover)

    with pytest.raises(TomeArtifactError) as exc_info:
        open_tome_artifact(artifact)

    assert any("cover_page_invalid" in item for item in exc_info.value.blockers)


def test_legacy_dense_fixture_is_explicit_smoke_debug_path(tmp_path: Path) -> None:
    from tome_fixtures import write_dense_tome

    view = open_tome_artifact(write_dense_tome(tmp_path / "legacy"))

    assert view.contract_family == "legacy_dense_v0"
    assert view.validation.student_interpretation == "legacy_smoke_debug_only"
    assert view.surfaces == ()
    assert view.recommended_training_plan is None
    assert "legacy_dense_v0_smoke_path" in view.warnings


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "artifact"
    shutil.copytree(production_tome_fixture_path(), destination)
    return destination


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _content_ref(cover: dict, role: str) -> dict:
    return next(item for item in cover["contents"] if item["role"] == role)
