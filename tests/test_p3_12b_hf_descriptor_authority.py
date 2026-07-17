from __future__ import annotations

from dataclasses import replace

import pytest

from radjax_student.architecture import (
    ArchitectureInitResult,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.contracts import (
    HFArchitectureProjection,
    HFCompatibilityDescriptor,
    HFContractError,
    HFParameterProjection,
    HFPreservationReference,
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
    hf_digest,
)
from radjax_student.learning import RunHFSummary
from radjax_student.validation.p3_12b_hf_descriptor_authority.models import (
    ADVERSARIAL_CASE_COUNT,
    validate_receipt,
)


def _descriptor() -> HFCompatibilityDescriptor:
    return HFCompatibilityDescriptor(
        "hf_compatibility_descriptor.v2",
        "test.hf_descriptor",
        1,
        "radjax_validation",
        hf_digest("config"),
        hf_digest("catalog"),
        hf_digest("layout"),
        HFTokenizerIdentity(
            "synthetic-tokenizer",
            "r1",
            hf_digest("tokenizer"),
            hf_digest("tokenizer-config"),
            "synthetic",
            hf_digest("normalization"),
            "synthetic",
        ),
        HFVocabularyIdentity(
            8, hf_digest("vocabulary"), hf_digest("tokens"), hf_digest("added"), None
        ),
        HFSpecialTokenIdentity(0, 1, 2, 3, None),
        (
            HFParameterProjection(
                "weight",
                ("weight",),
                (1,),
                "float32",
                "exportable",
                "model.weight",
                "identity",
            ),
        ),
        HFArchitectureProjection("synthetic", "linear", 1, 1, 8, 1, {}),
        ("hf_export_not_implemented",),
        "descriptive prose only",
    )


def test_reference_is_exact_descriptor_projection():
    descriptor = _descriptor()
    reference = descriptor.preservation_reference()
    assert reference == descriptor.preservation_reference()
    assert HFPreservationReference.from_dict(reference.to_dict()) == reference


def test_descriptor_parse_rejects_unknown_or_missing_fields():
    payload = _descriptor().to_dict()
    with pytest.raises(HFContractError, match="fields are invalid"):
        HFCompatibilityDescriptor.from_dict({**payload, "unknown": True})
    payload.pop("model_type")
    with pytest.raises(HFContractError, match="fields are invalid"):
        HFCompatibilityDescriptor.from_dict(payload)


def test_descriptor_digest_excludes_only_descriptive_prose():
    descriptor = _descriptor()
    assert (
        replace(descriptor, notes="different explanation").digest == descriptor.digest
    )
    assert replace(descriptor, model_type="foreign").digest != descriptor.digest


def test_exportability_requires_a_canonical_key_or_reason():
    with pytest.raises(HFContractError, match="requires exactly one HF key"):
        HFParameterProjection(
            "weight",
            ("weight",),
            (),
            "float32",
            "exportable",
            None,
            "identity",
        )
    with pytest.raises(HFContractError, match="requires a stable reason"):
        HFParameterProjection(
            "buffer",
            ("buffer",),
            (),
            "int32",
            "non_exportable",
            None,
            "identity",
        )


def test_report_carries_summary_not_full_descriptor():
    descriptor = _descriptor()
    summary = RunHFSummary(descriptor).to_dict()
    assert summary["descriptor_digest"] == descriptor.digest
    assert "parameter_projections" not in summary
    assert "tokenizer" not in summary


def test_initialization_rejects_reference_without_descriptor():
    descriptor = _descriptor()
    catalog = ParameterCatalog(
        "test.hf_descriptor", (ParameterDescriptor("weight", (1,), "float32"),)
    )
    layout = ParameterTreeLayout(
        "test.hf_descriptor",
        (
            ParameterTreeLayoutEntry(
                "weight",
                ("weight",),
                (1,),
                "float32",
                "other",
                ("whole_student",),
                exportable=True,
                hf_distribution_key="model.weight",
            ),
        ),
    )
    with pytest.raises(ValueError, match="hf_descriptor_missing"):
        ArchitectureInitResult(
            parameter_catalog=catalog,
            parameter_layout=layout,
            hf_reference=descriptor.preservation_reference(),
        )


def test_literal_gate_inventory_has_exactly_77_distinct_experiments():
    from radjax_student.validation.p3_12b_hf_descriptor_authority.runner_jax import (
        SPECS,
    )

    assert len(SPECS) == ADVERSARIAL_CASE_COUNT
    assert len({spec.case_id for spec in SPECS}) == ADVERSARIAL_CASE_COUNT
    assert len({spec.experiment for spec in SPECS}) == ADVERSARIAL_CASE_COUNT


def test_v2_receipt_rejects_incomplete_adversarial_inventory():
    import json
    from pathlib import Path

    payload = json.loads(
        Path("docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
    )
    assert validate_receipt(payload)["adversarial_case_count"] == 77
    payload["adversarial_case_count"] = 76
    with pytest.raises(ValueError, match="schema or status"):
        validate_receipt(payload)


@pytest.mark.jax
def test_lifecycle_rejects_independently_fabricated_reference():
    from radjax_student.validation.p3_11_9_replay.runner_jax import _new_lifecycle

    lifecycle = _new_lifecycle("eager", [])
    fabricated = HFPreservationReference.from_dict(
        {**lifecycle.hf_reference.to_dict(), "descriptor_digest": "0" * 64}
    )
    with pytest.raises(ValueError, match="not derived"):
        replace(lifecycle, hf_reference=fabricated)
