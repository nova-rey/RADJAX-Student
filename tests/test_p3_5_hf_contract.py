from __future__ import annotations

import hashlib

import pytest

from radjax_student.contracts import (
    HFArchitectureProjection,
    HFCompatibilityDescriptor,
    HFCompatibilityError,
    HFParameterProjection,
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _projection(
    path: str, keypath: tuple[str, ...], hf_key: str
) -> HFParameterProjection:
    return HFParameterProjection(
        logical_path=path,
        jax_keypath=keypath,
        shape=(4,),
        dtype="float32",
        exportability="exportable",
        hf_distribution_key=hf_key,
        projection_rule="identity",
    )


def _descriptor(**changes: object) -> HFCompatibilityDescriptor:
    values: dict[str, object] = {
        "schema_version": "hf_compatibility_descriptor.v2",
        "architecture_id": "linear",
        "architecture_plugin_version": 1,
        "model_type": "radjax-linear",
        "architecture_config_digest": _digest("config"),
        "parameter_catalog_digest": _digest("catalog"),
        "parameter_layout_digest": _digest("layout"),
        "tokenizer": HFTokenizerIdentity(
            "toy-tokenizer",
            "r1",
            _digest("tokenizer"),
            _digest("tokenizer-config"),
            "toy",
            _digest("normalization"),
            "synthetic",
        ),
        "vocabulary": HFVocabularyIdentity(
            8, _digest("vocabulary"), _digest("token-ids"), _digest("added"), None
        ),
        "special_tokens": HFSpecialTokenIdentity(None, 2, 0, None, None),
        "parameter_projections": (
            _projection("head.bias", ("head", "bias"), "classifier.bias"),
            _projection("head.weight", ("head", "weight"), "classifier.weight"),
        ),
        "architecture_projection": HFArchitectureProjection(
            "radjax", "linear", 4, 1, 8, 16, {"hidden_size": 4}
        ),
        "non_claims": ("hf_export_not_implemented",),
        "notes": "synthetic validation descriptor",
    }
    values.update(changes)
    return HFCompatibilityDescriptor(**values)  # type: ignore[arg-type]


def test_hf_descriptor_round_trip_preserves_authoritative_identity():
    descriptor = _descriptor()
    restored = HFCompatibilityDescriptor.from_dict(descriptor.to_dict())
    assert restored == descriptor
    assert restored.digest == descriptor.digest
    assert restored.preservation_reference() == descriptor.preservation_reference()


def test_hf_descriptor_rejects_unknown_fields():
    payload = {**_descriptor().to_dict(), "future_field": {"enabled": True}}
    with pytest.raises(HFCompatibilityError, match="fields are invalid"):
        HFCompatibilityDescriptor.from_dict(payload)


def test_hf_key_is_required_for_exportable_projection():
    with pytest.raises(HFCompatibilityError, match="requires exactly one HF key"):
        HFParameterProjection(
            "head.weight",
            ("head", "weight"),
            (4,),
            "float32",
            "exportable",
            None,
            "identity",
        )


def test_non_exportable_projection_requires_reason_and_no_hf_key():
    projection = HFParameterProjection(
        "runtime.buffer",
        ("runtime", "buffer"),
        (),
        "int32",
        "non_exportable",
        None,
        "identity",
        non_exportability_reason="runtime_only",
    )
    assert projection.hf_distribution_key is None
    with pytest.raises(HFCompatibilityError, match="requires a stable reason"):
        HFParameterProjection(
            "runtime.buffer",
            ("runtime", "buffer"),
            (),
            "int32",
            "non_exportable",
            None,
            "identity",
        )


def test_descriptor_rejects_duplicate_projection_paths_and_hf_keys():
    first, second = _descriptor().parameter_projections
    with pytest.raises(HFCompatibilityError, match="parameter paths must be unique"):
        _descriptor(parameter_projections=(first, first))
    duplicate_key = HFParameterProjection(
        second.logical_path,
        second.jax_keypath,
        second.shape,
        second.dtype,
        second.exportability,
        first.hf_distribution_key,
        second.projection_rule,
    )
    with pytest.raises(HFCompatibilityError, match="shared HF key requires"):
        _descriptor(parameter_projections=(first, duplicate_key))


def test_special_tokens_are_unique_and_within_vocabulary():
    with pytest.raises(HFCompatibilityError, match="assignments conflict"):
        _descriptor(special_tokens=HFSpecialTokenIdentity(None, 0, 0, None, None))
    with pytest.raises(HFCompatibilityError, match="outside vocabulary"):
        _descriptor(special_tokens=HFSpecialTokenIdentity(None, 8, 0, None, None))


def test_notes_do_not_change_descriptor_identity_but_identity_fields_do():
    descriptor = _descriptor()
    prose = _descriptor(notes="different human-facing explanation")
    assert descriptor.digest == prose.digest
    assert descriptor.to_dict() != prose.to_dict()
    changed = _descriptor(model_type="different-linear")
    assert descriptor.digest != changed.digest
