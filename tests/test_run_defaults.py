from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from radjax_contract.testing import production_tome_fixture_path
from tome_fixtures import write_dense_tome

from radjax_student.artifacts import TomeArtifactError, open_tome_artifact
from radjax_student.validation import (
    infer_run_defaults,
    infer_run_defaults_from_tome,
)

CURRENT_REQUIRED_CAPABILITIES = (
    "radjax.corridor.packed_assignments.v1",
    "radjax.corridor.stat_bands.v1",
    "radjax.exemplar.selected_dynamic_topk.v1",
)


def test_production_defaults_preserve_artifact_facts() -> None:
    defaults = infer_run_defaults_from_tome(production_tome_fixture_path())
    facts = defaults.artifact_facts

    assert facts.contract_family == "production_v2"
    assert facts.artifact_kind == "radjax_tome"
    assert facts.cover_page_version == 2
    assert facts.tome_version == 1
    assert facts.layout == "unpacked_directory"
    assert facts.source_artifact_type == "teacher_textbook"
    assert facts.artifact_id is None
    assert facts.teacher_model_identity == "fake-production-teacher"
    assert facts.teacher_model_revision == "fixture-v1"
    assert facts.teacher_family == "fake"
    assert facts.teacher_backend == "fake"
    assert facts.tokenizer_id == "fake-production-tokenizer"
    assert facts.tokenizer_hash == "sha256:fixture-tokenizer-v1"
    assert facts.vocab_size == 32
    assert facts.sequence_length == 4
    assert facts.example_count == 8
    assert facts.producer_validation_status == "pass"
    assert facts.contract_validation_status == "pass"
    assert facts.content_count == 21
    assert facts.surface_count == 2


def test_production_defaults_report_every_surface_and_typed_summary() -> None:
    defaults = infer_run_defaults(open_tome_artifact(production_tome_fixture_path()))

    assert [surface.surface_id for surface in defaults.available_surfaces] == [
        "corridor",
        "exemplar",
    ]
    corridor, exemplar = defaults.available_surfaces
    assert corridor.surface_kind == "fingerprint_corridor"
    assert corridor.known_surface
    assert corridor.target_scope == {"kind": "whole_model"}
    assert corridor.corridor is not None
    assert corridor.corridor.mode_policy == "stat_bands_v0"
    assert corridor.corridor.mode_count == 2
    assert corridor.corridor.assignment_count == 32
    assert corridor.corridor.assignment_storage_kind == "packed_numpy_v1"
    assert corridor.corridor.corridor_stat_top_k == 32
    assert not corridor.corridor.degraded
    assert exemplar.surface_kind == "selected_exemplar"
    assert exemplar.prerequisites == ("corridor",)
    assert exemplar.exemplar is not None
    assert exemplar.exemplar.selected_exemplar_count == 4
    assert exemplar.exemplar.payload_shard_count == 1
    assert exemplar.exemplar.corridor_linkage_required
    assert exemplar.exemplar.dynamic_top_k_metadata == {
        "dynamic_mass_threshold": "0.95",
        "dynamic_top_k_max": "5",
        "dynamic_top_k_min": "2",
    }


def test_capabilities_are_deterministic_requirements_not_compatibility() -> None:
    defaults = infer_run_defaults_from_tome(production_tome_fixture_path())

    assert defaults.required_capabilities == CURRENT_REQUIRED_CAPABILITIES
    assert defaults.unsupported_required_capabilities == CURRENT_REQUIRED_CAPABILITIES
    assert defaults.capabilities_not_yet_evaluated == CURRENT_REQUIRED_CAPABILITIES
    payload = defaults.to_dict()
    assert "compatible" not in payload
    assert "compatibility_status" not in payload
    assert "consumable" not in payload
    assert "compatibility_not_evaluated" in defaults.student_claims_not_made
    assert "required_capabilities_not_proven" in defaults.student_claims_not_made


def test_recommended_plan_remains_declarative_data() -> None:
    defaults = infer_run_defaults_from_tome(production_tome_fixture_path())
    plan = defaults.recommended_training_plan

    assert [training_pass.pass_index for training_pass in plan] == [0, 1]
    assert [training_pass.surface_id for training_pass in plan] == [
        "corridor",
        "exemplar",
    ]
    assert [training_pass.checkpoint_after for training_pass in plan] == [True, True]
    assert plan[0].prerequisites == ()
    assert plan[1].prerequisites == ("corridor",)
    assert plan[0].target_scope == {"kind": "whole_model"}
    assert plan[1].target_scope == {"kind": "whole_model"}
    assert not hasattr(defaults, "schedule")
    assert not hasattr(defaults, "optimizer")


def test_user_choices_and_later_policy_remain_unresolved() -> None:
    defaults = infer_run_defaults_from_tome(production_tome_fixture_path())

    assert defaults.required_from_user == {
        "student_architecture": None,
        "student_size_or_config": None,
        "training_budget": None,
        "output_dir": None,
    }
    assert defaults.unresolved_by_phase == {
        "runtime_backend": "phase_2_or_later",
        "precision_policy": "phase_2_or_later",
        "optimizer": "phase_3_or_later",
        "schedule_implementation": "phase_3_or_later",
        "loss_weighting": "phase_3_or_later",
        "architecture_plugin": "phase_4",
        "evaluation_policy": "phase_5_or_later",
        "hf_export_details": "phase_5_or_later",
    }
    assert "teacher_id" not in defaults.required_from_user
    assert "tokenizer_id" not in defaults.required_from_user
    assert "sequence_length" not in defaults.required_from_user


def test_artifact_and_student_claims_remain_separate_and_json_safe() -> None:
    defaults = infer_run_defaults_from_tome(production_tome_fixture_path())
    payload = defaults.to_dict()

    assert payload["artifact_claims_not_made"] == [
        "no_model_quality_claim",
        "no_network_verification_claim",
        "no_student_training_claim",
    ]
    assert "training_not_run" in payload["student_claims_not_made"]
    assert "payload_tensors_not_loaded" in payload["student_claims_not_made"]
    assert "model_not_allocated" in payload["student_claims_not_made"]
    assert "artifact_claims_not_made" in payload
    assert "student_claims_not_made" in payload
    assert "claims_not_made" not in payload
    assert payload["legacy_smoke_defaults"] is None
    json.dumps(payload)


def test_production_defaults_are_immutable() -> None:
    defaults = infer_run_defaults_from_tome(production_tome_fixture_path())

    with pytest.raises(TypeError):
        defaults.required_from_user["training_budget"] = 10  # type: ignore[index]
    with pytest.raises(TypeError):
        defaults.available_surfaces[0].target_scope["kind"] = "changed"  # type: ignore[index]


def test_unknown_surface_and_required_capability_are_preserved(
    tmp_path: Path,
) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    future_capability = "radjax.future.required.v1"
    cover["behavioral_surfaces"].append(
        {
            "optional_content_roles": [],
            "prerequisites": [],
            "required_capabilities": [future_capability],
            "required_content_roles": [],
            "schema_version": "future_surface_v1",
            "semantics": {"future_semantic": "preserved"},
            "surface_id": "future_optional",
            "surface_kind": "future_optional_kind",
            "target_scope": {"kind": "plugin_defined", "plugin": "future"},
        }
    )
    cover["recommended_training_plan"]["passes"].append(
        {
            "checkpoint_after": False,
            "pass_id": "future_pass",
            "prerequisites": [],
            "required_capabilities": [future_capability],
            "surface_id": "future_optional",
        }
    )
    _write_json(artifact / "cover_page.json", cover)

    defaults = infer_run_defaults_from_tome(artifact)

    future = defaults.available_surfaces[2]
    assert future.surface_id == "future_optional"
    assert future.surface_kind == "future_optional_kind"
    assert not future.known_surface
    assert future.target_scope == {"kind": "plugin_defined", "plugin": "future"}
    assert future.semantics == {"future_semantic": "preserved"}
    assert future_capability in defaults.required_capabilities
    assert future_capability in defaults.unsupported_required_capabilities
    assert future_capability in defaults.capabilities_not_yet_evaluated
    assert defaults.recommended_training_plan[2].surface_id == "future_optional"
    assert not defaults.recommended_training_plan[2].checkpoint_after
    assert "compatibility_not_evaluated" in defaults.student_claims_not_made
    json.dumps(defaults.to_dict())


def test_legacy_dense_defaults_remain_explicit_smoke_only(tmp_path: Path) -> None:
    defaults = infer_run_defaults_from_tome(write_dense_tome(tmp_path / "legacy"))

    assert defaults.artifact_facts.contract_family == "legacy_dense_v0"
    assert defaults.available_surfaces == ()
    assert defaults.required_capabilities == ()
    assert defaults.recommended_training_plan == ()
    assert defaults.legacy_smoke_defaults is not None
    assert defaults.legacy_smoke_defaults["classification"] == (
        "legacy_dense_v0_smoke_only"
    )
    assert defaults.inferred_from_tome["payload_format"] == "dense_logits_v0"
    assert defaults.inferred_from_tome["expected_adapter_family"] == "dense_logits"
    assert "legacy_dense_v0_smoke_path" in defaults.warnings
    json.dumps(defaults.to_dict())


def test_run_defaults_path_helper_preserves_artifact_errors(tmp_path: Path) -> None:
    tome = write_dense_tome(tmp_path / "tome")
    (tome / "cover_page.json").unlink()

    with pytest.raises(TomeArtifactError):
        infer_run_defaults_from_tome(tome)


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
