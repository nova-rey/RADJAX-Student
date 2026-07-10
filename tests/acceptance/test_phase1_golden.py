from __future__ import annotations

import json
import tomllib
from dataclasses import replace

import pytest
from radjax_contract.tome.production import (
    ArtifactLocalFingerprintId,
    ArtifactLocalModeId,
    validate_production_tome,
)

from radjax_student.artifacts import open_tome_artifact
from radjax_student.reports import artifact_tree_digest
from radjax_student.validation import (
    declaration_test_only_profile,
    evaluate_student_compatibility,
    infer_run_defaults,
    metadata_inspection_only_profile,
)

from .support import (
    ACCEPTED_FIXTURE_DIGEST,
    REPO_ROOT,
    REQUIRED_CAPABILITIES,
    add_unknown_optional_content,
    canonical_fixture,
    finding_codes,
    normalized_doctor_payload,
    normalized_inspection_payload,
    read_golden,
    rename_indexed_content,
)

CONTRACT_RECEIPT_COMMIT = "ff8f6e9af976fc599ee31173d4f177fb1250b4d7"


def test_canonical_fixture_and_dependency_pin_match_accepted_receipt() -> None:
    fixture = canonical_fixture()
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    contract_requirement = next(
        item
        for item in project["project"]["dependencies"]
        if item.startswith("radjax-contract ")
    )

    assert fixture.is_dir()
    assert artifact_tree_digest(fixture) == ACCEPTED_FIXTURE_DIGEST
    assert contract_requirement.endswith(f"@{CONTRACT_RECEIPT_COMMIT}")
    assert validate_production_tome(fixture).status == "pass"


def test_artifact_view_preserves_full_production_contract() -> None:
    fixture = canonical_fixture()
    view = open_tome_artifact(fixture)

    assert not (fixture / "manifest.json").exists()
    assert view.contract_family == "production_v2"
    assert view.identity.artifact_kind == "radjax_tome"
    assert view.identity.cover_page_version == 2
    assert view.identity.tome_version == 1
    assert view.provenance.teacher["model_id"] == "fake-production-teacher"
    assert view.provenance.tokenizer["tokenizer_id"] == "fake-production-tokenizer"
    assert view.validation.producer_status == "pass"
    assert view.validation.contract_status == "pass"
    assert view.claims_not_made
    assert len(view.contents_index) == 21
    assert [surface.surface_id for surface in view.surfaces] == [
        "corridor",
        "exemplar",
    ]
    assert view.corridor_contract is not None
    assert view.exemplar_contract is not None
    assert view.validation.required_capabilities == REQUIRED_CAPABILITIES
    assert view.recommended_training_plan is not None
    assert [
        item.checkpoint_after for item in view.recommended_training_plan.passes
    ] == [True, True]
    assert not hasattr(view.corridor_contract, "assignments")
    assert not hasattr(view.corridor_contract, "fingerprints")
    assert not hasattr(view.exemplar_contract, "payloads")
    assert ArtifactLocalModeId.from_value(1) != ArtifactLocalFingerprintId.from_value(1)


def test_artifact_view_uses_content_index_without_directory_walk_or_name_guessing(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    from shutil import copytree

    copytree(canonical_fixture(), artifact)
    rename_indexed_content(artifact)

    def forbid_walk(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Student artifact opening walked the fixture directory")

    monkeypatch.setattr(type(artifact), "rglob", forbid_walk)
    view = open_tome_artifact(artifact)

    assert view.corridor_contract is not None
    assert any(ref.path == "renamed/summary.data" for ref in view.contents_index)


def test_run_defaults_are_complete_declarative_and_deterministic() -> None:
    view = open_tome_artifact(canonical_fixture())
    first = infer_run_defaults(view)
    second = infer_run_defaults(view)
    payload = first.to_dict()

    assert first.artifact_facts.artifact_kind == "radjax_tome"
    assert [surface.surface_id for surface in first.available_surfaces] == [
        "corridor",
        "exemplar",
    ]
    assert first.required_capabilities == REQUIRED_CAPABILITIES
    assert [item.surface_id for item in first.recommended_training_plan] == [
        "corridor",
        "exemplar",
    ]
    assert all(value is None for value in first.required_from_user.values())
    assert first.unresolved_by_phase
    assert first.artifact_claims_not_made
    assert first.student_claims_not_made
    assert first.legacy_smoke_defaults is None
    assert "compatible" not in payload
    assert "compatibility_status" not in payload
    assert "payload_format" not in payload["artifact_facts"]
    assert "adapter" not in payload["artifact_facts"]
    assert not hasattr(first, "schedule")
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        second.to_dict(),
        sort_keys=True,
    )


def test_unknown_optional_surface_survives_defaults_without_verdict(tmp_path) -> None:
    from .support import copy_fixture

    artifact = copy_fixture(tmp_path)
    add_unknown_optional_content(artifact)

    defaults = infer_run_defaults(open_tome_artifact(artifact))
    future = defaults.available_surfaces[-1]

    assert future.surface_id == "future_optional"
    assert not future.known_surface
    assert future.target_scope["kind"] == "plugin_defined"
    assert "compatibility" not in defaults.to_dict()


def test_metadata_profile_fails_honestly_and_reproducibly() -> None:
    view = open_tome_artifact(canonical_fixture())
    defaults = infer_run_defaults(view)
    first = evaluate_student_compatibility(
        view,
        defaults,
        metadata_inspection_only_profile(),
    )
    second = evaluate_student_compatibility(
        view,
        defaults,
        metadata_inspection_only_profile(),
    )

    assert first.status == "fail"
    assert first.missing_capabilities == REQUIRED_CAPABILITIES
    assert finding_codes(first.blockers).count("missing_required_capability") == 3
    assert first.sequence_compatibility.status == "unevaluated"
    assert first.vocab_compatibility.status == "unevaluated"
    assert first.to_dict() == second.to_dict()


def test_declaration_profile_passes_logic_without_execution_claim() -> None:
    view = open_tome_artifact(canonical_fixture())
    report = evaluate_student_compatibility(
        view,
        infer_run_defaults(view),
        declaration_test_only_profile(),
    )

    assert report.status == "pass"
    assert "capability_declaration_not_execution_proof" in finding_codes(
        report.warnings
    )
    assert "capability_declaration_not_implementation_proof" in (
        report.student_claims_not_made
    )


@pytest.mark.parametrize(
    ("profile_change", "expected_code"),
    [
        ({"supported_contract_families": ()}, "unsupported_contract_family"),
        ({"supported_tome_versions": ()}, "unsupported_tome_version"),
        ({"supported_surface_kinds": ()}, "unsupported_surface_kind"),
        ({"supported_surface_schemas": ()}, "unsupported_surface_schema"),
        (
            {"supported_capabilities": REQUIRED_CAPABILITIES[:-1]},
            "missing_required_capability",
        ),
        ({"max_sequence_length": 3}, "sequence_length_exceeds_profile"),
        ({"max_vocab_size": 31}, "vocab_size_exceeds_profile"),
        ({"supported_target_scopes": ()}, "unsupported_target_scope"),
        ({"supports_checkpoint_boundaries": False}, "unsupported_training_plan"),
        ({"accepted_producer_statuses": ()}, "producer_validation_failed"),
    ],
)
def test_compatibility_blocker_codes_are_stable(
    profile_change: dict,
    expected_code: str,
) -> None:
    view = open_tome_artifact(canonical_fixture())
    defaults = infer_run_defaults(view)
    profile = replace(declaration_test_only_profile(), **profile_change)

    report = evaluate_student_compatibility(view, defaults, profile)

    assert report.status == "fail"
    assert expected_code in finding_codes(report.blockers)


def test_contract_validation_failure_code_is_stable() -> None:
    view = open_tome_artifact(canonical_fixture())
    defaults = infer_run_defaults(view)
    altered = replace(
        defaults,
        artifact_facts=replace(
            defaults.artifact_facts,
            contract_validation_status="fail",
        ),
    )

    report = evaluate_student_compatibility(
        view,
        altered,
        declaration_test_only_profile(),
    )

    assert report.status == "fail"
    assert "contract_validation_failed" in finding_codes(report.blockers)


def test_normalized_golden_reports_match_checked_in_artifacts() -> None:
    assert normalized_inspection_payload() == read_golden(
        "phase1_inspect_metadata_only.json"
    )
    assert normalized_doctor_payload() == read_golden("phase1_doctor.json")
