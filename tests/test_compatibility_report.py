from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from radjax_contract.testing import production_tome_fixture_path

from radjax_student.artifacts import TomeArtifactError, open_tome_artifact
from radjax_student.validation import (
    StudentCapabilityProfile,
    evaluate_student_compatibility,
    evaluate_tome_path_compatibility,
    infer_run_defaults,
    metadata_inspection_only_profile,
)

REQUIRED_CAPABILITIES = (
    "radjax.corridor.packed_assignments.v1",
    "radjax.corridor.stat_bands.v1",
    "radjax.exemplar.selected_dynamic_topk.v1",
)


def test_metadata_only_profile_fails_honestly() -> None:
    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        metadata_inspection_only_profile(),
    )

    assert report.status == "fail"
    assert not report.ok
    assert report.missing_capabilities == REQUIRED_CAPABILITIES
    assert _codes(report.blockers).count("missing_required_capability") == 3
    assert "sequence_compatibility_unevaluated" in _codes(report.blockers)
    assert "vocab_compatibility_unevaluated" in _codes(report.blockers)
    assert report.sequence_compatibility.status == "unevaluated"
    assert report.vocab_compatibility.status == "unevaluated"
    assert report.supported_surfaces == ("corridor", "exemplar")


def test_full_synthetic_declared_profile_passes_evaluator_logic() -> None:
    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        _full_profile(),
    )

    assert report.status == "pass"
    assert report.ok
    assert report.blockers == ()
    assert report.required_capabilities == REQUIRED_CAPABILITIES
    assert report.supported_capabilities == REQUIRED_CAPABILITIES
    assert report.missing_capabilities == ()
    assert report.unevaluated_capabilities == REQUIRED_CAPABILITIES
    assert report.supported_surfaces == ("corridor", "exemplar")
    assert report.unsupported_surfaces == ()
    assert report.sequence_compatibility.status == "pass"
    assert report.vocab_compatibility.status == "pass"
    assert report.plan_compatibility.status == "pass"
    assert "capability_declaration_not_execution_proof" in _codes(report.warnings)
    assert "architecture_specific_compatibility_not_evaluated" in _codes(
        report.warnings
    )


def test_missing_one_capability_yields_one_stable_blocker() -> None:
    profile = replace(
        _full_profile(),
        supported_capabilities=REQUIRED_CAPABILITIES[:-1],
    )

    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        profile,
    )

    missing = [
        blocker
        for blocker in report.blockers
        if blocker.code == "missing_required_capability"
    ]
    assert report.status == "fail"
    assert report.missing_capabilities == (REQUIRED_CAPABILITIES[-1],)
    assert len(missing) == 1
    assert missing[0].details["capability"] == REQUIRED_CAPABILITIES[-1]


def test_unknown_optional_surface_warns_without_blocking(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    cover["behavioral_surfaces"].append(_future_surface())
    cover["behavioral_surfaces"].append(_future_surface("future_surface_two"))
    _write_json(artifact / "cover_page.json", cover)

    report = evaluate_tome_path_compatibility(artifact, _full_profile())

    assert report.status == "pass"
    assert _codes(report.warnings).count("unknown_optional_surface") == 2
    assert report.unsupported_surfaces == ("future_surface", "future_surface_two")
    assert report.target_scope_compatibility[-2].status == "unevaluated"
    assert report.target_scope_compatibility[-1].status == "unevaluated"


def test_unknown_required_surface_blocks(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    cover["behavioral_surfaces"].append(_future_surface())
    cover["recommended_training_plan"]["passes"].append(
        {
            "checkpoint_after": False,
            "pass_id": "future_pass",
            "prerequisites": [],
            "required_capabilities": [],
            "surface_id": "future_surface",
        }
    )
    _write_json(artifact / "cover_page.json", cover)

    report = evaluate_tome_path_compatibility(artifact, _full_profile())

    assert report.status == "fail"
    assert "unknown_required_surface" in _codes(report.blockers)
    assert "future_surface" in report.unsupported_surfaces


def test_unsupported_target_scope_blocks_required_surface(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    cover["behavioral_surfaces"][0]["target_scope"] = {
        "kind": "named_architecture_region",
        "region": "attention",
    }
    _write_json(artifact / "cover_page.json", cover)

    report = evaluate_tome_path_compatibility(artifact, _full_profile())

    assert report.status == "fail"
    assert "unsupported_target_scope" in _codes(report.blockers)
    corridor_scope = report.target_scope_compatibility[0]
    assert corridor_scope.scope_kind == "named_architecture_region"
    assert corridor_scope.status == "fail"


def test_sequence_limit_failure_is_explicit() -> None:
    profile = replace(_full_profile(), max_sequence_length=3)

    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        profile,
    )

    assert report.status == "fail"
    assert "sequence_length_exceeds_profile" in _codes(report.blockers)
    assert report.sequence_compatibility.artifact_value == 4
    assert report.sequence_compatibility.profile_limit == 3


def test_vocab_limit_failure_is_explicit() -> None:
    profile = replace(_full_profile(), max_vocab_size=31)

    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        profile,
    )

    assert report.status == "fail"
    assert "vocab_size_exceeds_profile" in _codes(report.blockers)
    assert report.vocab_compatibility.artifact_value == 32
    assert report.vocab_compatibility.profile_limit == 31


def test_tokenizer_support_is_explicit() -> None:
    profile = replace(
        _full_profile(),
        supported_tokenizer_ids=("different-tokenizer",),
    )

    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        profile,
    )

    assert report.status == "fail"
    assert "unsupported_tokenizer" in _codes(report.blockers)


def test_surface_schema_support_is_explicit() -> None:
    profile = replace(
        _full_profile(),
        supported_surface_schemas=(("fingerprint_corridor", "behavioral_surface_v1"),),
    )

    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        profile,
    )

    assert report.status == "fail"
    assert "unsupported_surface_schema" in _codes(report.blockers)
    assert report.unsupported_surfaces == ("exemplar",)


@pytest.mark.parametrize(
    ("field", "value", "blocker_code"),
    [
        ("supported_contract_families", (), "unsupported_contract_family"),
        ("supported_tome_versions", (), "unsupported_tome_version"),
        ("supported_cover_page_versions", (), "unsupported_cover_page_version"),
        ("accepted_producer_statuses", (), "producer_validation_failed"),
    ],
)
def test_identity_and_producer_policy_are_explicit(
    field: str,
    value: tuple,
    blocker_code: str,
) -> None:
    profile = replace(_full_profile(), **{field: value})

    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        profile,
    )

    assert report.status == "fail"
    assert blocker_code in _codes(report.blockers)


def test_checkpoint_plan_support_is_metadata_only() -> None:
    profile = replace(_full_profile(), supports_checkpoint_boundaries=False)

    report = evaluate_tome_path_compatibility(
        production_tome_fixture_path(),
        profile,
    )

    assert report.status == "fail"
    assert "unsupported_training_plan" in _codes(report.blockers)
    blocker = next(
        item for item in report.blockers if item.code == "unsupported_training_plan"
    )
    assert blocker.details["feature"] == "checkpoint_boundaries"
    assert report.plan_compatibility.status == "fail"
    assert not hasattr(report, "schedule")


def test_invalid_plan_reference_remains_owned_by_contract(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    cover["recommended_training_plan"]["passes"][0]["surface_id"] = "missing"
    _write_json(artifact / "cover_page.json", cover)

    with pytest.raises(TomeArtifactError) as exc_info:
        evaluate_tome_path_compatibility(artifact, _full_profile())

    assert any(
        "training_pass_surface_missing" in blocker
        for blocker in exc_info.value.blockers
    )


def test_report_preserves_claim_separation_and_serializes() -> None:
    view = open_tome_artifact(production_tome_fixture_path())
    defaults = infer_run_defaults(view)
    report = evaluate_student_compatibility(view, defaults, _full_profile())
    payload = report.to_dict()

    assert payload["status"] == "pass"
    assert payload["artifact_claims_not_made"] == [
        "no_model_quality_claim",
        "no_network_verification_claim",
        "no_student_training_claim",
    ]
    assert (
        "capability_declaration_not_implementation_proof"
        in payload["student_claims_not_made"]
    )
    assert "payload_loading_not_tested" in payload["student_claims_not_made"]
    assert "checkpoint_execution_not_tested" in payload["student_claims_not_made"]
    assert not hasattr(report, "model")
    assert not hasattr(report, "runtime")
    json.dumps(payload)


def test_profile_is_immutable_and_json_serializable() -> None:
    profile = _full_profile()

    with pytest.raises(AttributeError):
        profile.profile_id = "changed"  # type: ignore[misc]
    json.dumps(profile.to_dict())


def _full_profile() -> StudentCapabilityProfile:
    return StudentCapabilityProfile(
        profile_id="synthetic_full_declaration",
        supported_contract_families=("production_v2",),
        supported_tome_versions=(1,),
        supported_cover_page_versions=(2,),
        supported_surface_kinds=("fingerprint_corridor", "selected_exemplar"),
        supported_surface_schemas=(
            ("fingerprint_corridor", "behavioral_surface_v1"),
            ("selected_exemplar", "behavioral_surface_v1"),
        ),
        supported_capabilities=REQUIRED_CAPABILITIES,
        supported_target_scopes=("whole_model", "unspecified", "default"),
        max_sequence_length=4,
        max_vocab_size=32,
        supported_tokenizer_ids=("fake-production-tokenizer",),
        notes=("synthetic declaration for evaluator tests only",),
    )


def _future_surface(surface_id: str = "future_surface") -> dict:
    return {
        "optional_content_roles": [],
        "prerequisites": [],
        "required_capabilities": [],
        "required_content_roles": [],
        "schema_version": "future_surface_v1",
        "semantics": {"future_semantic": "preserved"},
        "surface_id": surface_id,
        "surface_kind": "future_surface_kind",
        "target_scope": {"kind": "plugin_defined", "plugin": "future"},
    }


def _codes(findings: tuple) -> list[str]:
    return [item.code for item in findings]


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
