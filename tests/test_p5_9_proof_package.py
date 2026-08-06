"""P5.9 deterministic HF-shaped proof-package evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from radjax_student.architecture import ArchitectureConfig
from radjax_student.behavior import (
    HeldOutEvaluationBindingV1,
    HfShapedProofPackageError,
    architecture_config_identity_v1,
    build_hf_shaped_proof_package_v1,
    evaluate_held_out_behavior_v1,
    hf_language_projection_identity_v1,
    load_hf_shaped_proof_package_v1,
    replay_hf_shaped_proof_package_v1,
)
from radjax_student.contracts.hf import canonical_hf_json
from tests.test_p5_6_corridor_pass import _setup
from tests.test_p5_8_held_out_evaluation import _evaluation_values


def _proof_inputs():
    values = _evaluation_values()
    config = ArchitectureConfig("neutral.proof", vocab_size=4, sequence_length=3)
    projection = _setup()["projection"]
    source = values["corridor_checkpoint"].binding
    corridor_binding = replace(
        source,
        language_binding_digest=projection.language.canonical_binding_digest,
        hf_projection_identity=hf_language_projection_identity_v1(projection.language),
        behavioral_source_identity=projection.behavioral_source_identity,
        behavioral_authority_digest=projection.behavioral_authority_digest,
        architecture_config_identity=architecture_config_identity_v1(config),
    )
    corridor = replace(values["corridor_checkpoint"], binding=corridor_binding)
    final_binding = replace(
        values["final_checkpoint"].binding,
        predecessor_checkpoint_identity=corridor.identity,
        language_binding_digest=corridor_binding.language_binding_digest,
        hf_projection_identity=corridor_binding.hf_projection_identity,
        behavioral_source_identity=corridor_binding.behavioral_source_identity,
        behavioral_authority_digest=corridor_binding.behavioral_authority_digest,
        architecture_config_identity=corridor_binding.architecture_config_identity,
    )
    final = replace(values["final_checkpoint"], binding=final_binding)
    report_values = {
        **values,
        "corridor_checkpoint": corridor,
        "final_checkpoint": final,
        "binding": HeldOutEvaluationBindingV1.from_final_checkpoint(
            final, values["expected_batches"]
        ),
    }
    report = evaluate_held_out_behavior_v1(**report_values)
    return dict(
        projection=projection,
        architecture_config=config,
        corridor_checkpoint=corridor,
        final_checkpoint=final,
        evaluation_report=report,
    )


def test_p5_9_package_is_byte_deterministic_and_contains_only_declared_json():
    values = _proof_inputs()
    package = build_hf_shaped_proof_package_v1(**values)
    replay = replay_hf_shaped_proof_package_v1(expected=package, **values)
    assert replay.to_bytes() == package.to_bytes()
    assert tuple(name for name, _ in package.files) == tuple(
        sorted(name for name, _ in package.files)
    )
    assert tuple(name for name, _ in package.files) == (
        "config.json",
        "held_out_evaluation.json",
        "lineage.json",
        "nonclaims.json",
        "optimizer_identity.json",
        "provenance.json",
        "special_tokens_identity.json",
        "tokenizer_identity.json",
        "vocabulary_identity.json",
    )
    assert len(package.inventory) == 9
    assert tuple(name for name, _, _ in package.inventory) == tuple(
        name for name, _ in package.files
    )
    payloads = {name: json.loads(contents) for name, contents in package.files}
    assert set(payloads["config.json"]) == {
        "architectures",
        "radjax_architecture_config",
    }
    assert set(payloads["config.json"]["radjax_architecture_config"]) == {
        "architecture_id",
        "schema_version",
        "model_config",
        "vocab_size",
        "sequence_length",
        "dtype_intent",
        "metadata",
    }
    assert set(payloads["provenance.json"]) == {
        "hf_language_projection_version",
        "language_binding_digest",
        "hf_projection_identity",
        "contract_commit",
        "tome_commit",
        "fixture_receipt_identity",
        "behavioral_source_identity",
        "behavioral_authority_digest",
        "package_semantic_identity",
        "composition_digest",
    }
    assert set(payloads["lineage.json"]) == {
        "corridor_checkpoint_identity",
        "final_checkpoint_identity",
        "split_identity",
        "materialization_identity",
        "canonical_passport_registry_identity",
        "corridor_policy",
        "reduction",
        "exemplar_policy",
    }
    assert set(payloads["optimizer_identity.json"]) == {
        "optimizer_id",
        "optimizer_descriptor",
        "parameter_layout",
    }
    assert set(payloads["held_out_evaluation.json"]) == {
        "identity",
        "binding",
        "corridor_coordinates",
        "exemplar_passport_keys",
        "corridor_metrics",
        "exemplar_metrics",
    }
    assert set(payloads["nonclaims.json"]) == {"nonclaims"}
    assert set(payloads["tokenizer_identity.json"]) == {
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_content_digest",
        "tokenizer_config_digest",
        "tokenizer_family",
        "normalization_digest",
        "identity_availability",
    }
    assert set(payloads["vocabulary_identity.json"]) == {
        "vocabulary_size",
        "vocabulary_content_digest",
        "token_to_id_digest",
        "added_token_digest",
        "reserved_token_range",
    }
    assert set(payloads["special_tokens_identity.json"]) == {
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "mask_token_id",
        "additional_special_token_ids",
    }
    assert {name for name, _ in package.files} == {
        "config.json",
        "held_out_evaluation.json",
        "lineage.json",
        "nonclaims.json",
        "optimizer_identity.json",
        "provenance.json",
        "special_tokens_identity.json",
        "tokenizer_identity.json",
        "vocabulary_identity.json",
    }
    restored = load_hf_shaped_proof_package_v1(
        package.to_bytes(), expected_identity=package.package_identity
    )
    assert restored.to_bytes() == package.to_bytes()


@pytest.mark.parametrize(
    "field", ("behavioral_source_identity", "architecture_config_identity")
)
def test_p5_9_rejects_mismatched_factory_lineage(field):
    values = _proof_inputs()
    final = values["final_checkpoint"]
    values["final_checkpoint"] = replace(
        final, binding=replace(final.binding, **{field: "sha256:" + "0" * 64})
    )
    with pytest.raises(HfShapedProofPackageError, match="lineage mismatch"):
        build_hf_shaped_proof_package_v1(**values)


def test_p5_9_rejects_altered_bytes_and_nonattested_projection():
    values = _proof_inputs()
    package = build_hf_shaped_proof_package_v1(**values)
    altered = package.to_bytes().replace(b"not_phase_6", b"phase_6")
    with pytest.raises(HfShapedProofPackageError, match="altered|invalid"):
        load_hf_shaped_proof_package_v1(
            altered, expected_identity=package.package_identity
        )
    values["projection"] = replace(values["projection"])
    with pytest.raises(HfShapedProofPackageError, match="factory provenance"):
        build_hf_shaped_proof_package_v1(**values)


@pytest.mark.parametrize(
    "file_name,key",
    (
        ("provenance.json", "language_binding_digest"),
        ("provenance.json", "hf_projection_identity"),
        ("provenance.json", "contract_commit"),
        ("provenance.json", "tome_commit"),
        ("provenance.json", "fixture_receipt_identity"),
        ("provenance.json", "behavioral_source_identity"),
        ("provenance.json", "behavioral_authority_digest"),
        ("provenance.json", "package_semantic_identity"),
        ("provenance.json", "composition_digest"),
        ("lineage.json", "corridor_checkpoint_identity"),
        ("lineage.json", "final_checkpoint_identity"),
        ("lineage.json", "split_identity"),
        ("lineage.json", "materialization_identity"),
        ("lineage.json", "canonical_passport_registry_identity"),
        ("lineage.json", "corridor_policy"),
        ("lineage.json", "reduction"),
        ("lineage.json", "exemplar_policy"),
        ("optimizer_identity.json", "optimizer_id"),
        ("held_out_evaluation.json", "identity"),
    ),
)
def test_p5_9_loader_rejects_every_major_serialized_identity_mutation(file_name, key):
    package = build_hf_shaped_proof_package_v1(**_proof_inputs())
    envelope = json.loads(package.to_bytes())
    file_payload = json.loads(envelope["files"][file_name])
    file_payload[key] = "altered.v1"
    envelope["files"][file_name] = canonical_hf_json(file_payload).decode("utf-8")
    altered = canonical_hf_json(envelope)
    with pytest.raises(HfShapedProofPackageError, match="inventory|altered|invalid"):
        load_hf_shaped_proof_package_v1(
            altered, expected_identity=package.package_identity
        )


@pytest.mark.parametrize(
    "field",
    (
        "contract_commit",
        "tome_commit",
        "accepted_receipt_identity",
        "language_binding_digest",
        "hf_projection_identity",
        "behavioral_source_identity",
        "behavioral_authority_digest",
        "architecture_config_identity",
        "split_identity",
        "materialization_identity",
        "canonical_passport_registry_identity",
        "corridor_objective_policy_id",
        "reduction_id",
        "corridor_ordering_policy_id",
        "corridor_batching_policy_id",
        "exemplar_objective_policy_id",
        "exemplar_ordering_policy_id",
        "exemplar_batching_policy_id",
    ),
)
def test_p5_9_build_rejects_every_report_provenance_policy_or_identity_mutation(field):
    values = _proof_inputs()
    report = values["evaluation_report"]
    values["evaluation_report"] = replace(
        report, binding=replace(report.binding, **{field: "altered.v1"})
    )
    with pytest.raises(HfShapedProofPackageError, match="lineage mismatch"):
        build_hf_shaped_proof_package_v1(**values)
