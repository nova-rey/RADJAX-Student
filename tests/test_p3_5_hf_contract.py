from __future__ import annotations

import ast
import json

import pytest

from radjax_student.architecture import (
    ArchitectureConfig,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.hf import (
    HFCompatibilityDescriptor,
    HFCompatibilityError,
    HFParameterMapping,
)


def _config() -> ArchitectureConfig:
    return ArchitectureConfig(
        architecture_id="linear",
        model_config={"hidden_size": 4},
        vocab_size=8,
        sequence_length=16,
        dtype_intent="float32",
    )


def _catalog() -> ParameterCatalog:
    return ParameterCatalog(
        architecture_id="linear",
        parameters=(
            ParameterDescriptor("head.bias", (4,), "float32"),
            ParameterDescriptor("head.weight", (4, 8), "float32"),
        ),
    )


def _mappings() -> tuple[HFParameterMapping, ...]:
    return (
        HFParameterMapping(
            "head.bias", "head/bias", "classifier.bias", (4,), "float32"
        ),
        HFParameterMapping(
            "head.weight", "head/weight", "classifier.weight", (4, 8), "float32"
        ),
    )


def _descriptor(**changes) -> HFCompatibilityDescriptor:
    values = {
        "model_type": "radjax-linear",
        "tokenizer_id": "toy-tokenizer",
        "special_token_ids": {"pad": 0, "eos": 2},
        "parameter_mappings": _mappings(),
        "architecture_state_metadata": {"state_schema": "v1"},
    }
    values.update(changes)
    return HFCompatibilityDescriptor.from_architecture(_config(), _catalog(), **values)


def test_hf_descriptor_round_trip_preserves_identity_and_unknown_fields():
    payload = {**_descriptor().to_dict(), "future_field": {"enabled": True}}
    restored = HFCompatibilityDescriptor.from_dict(payload)
    assert restored.to_dict()["future_field"] == {"enabled": True}
    assert restored.to_json() == json.dumps(
        restored.to_dict(), sort_keys=True, separators=(",", ":")
    )
    restored.validate_against(_config(), _catalog())


def test_hf_descriptor_requires_nonempty_mapping_tuple():
    with pytest.raises(HFCompatibilityError, match="nonempty"):
        _descriptor(parameter_mappings=())


def test_hf_descriptor_rejects_duplicate_logical_paths():
    mapping = _mappings()[0]
    with pytest.raises(HFCompatibilityError, match="logical_path"):
        _descriptor(parameter_mappings=(mapping, mapping))


def test_hf_descriptor_rejects_duplicate_jax_paths():
    first, second = _mappings()
    duplicate = HFParameterMapping(
        second.logical_path,
        first.jax_pytree_path,
        second.hf_distribution_key,
        second.shape,
        second.dtype,
    )
    with pytest.raises(HFCompatibilityError, match="jax_pytree_path"):
        _descriptor(parameter_mappings=(first, duplicate))


def test_hf_descriptor_rejects_duplicate_hf_distribution_keys():
    first, second = _mappings()
    duplicate = HFParameterMapping(
        second.logical_path,
        second.jax_pytree_path,
        first.hf_distribution_key,
        second.shape,
        second.dtype,
    )
    with pytest.raises(HFCompatibilityError, match="hf_distribution_key"):
        _descriptor(parameter_mappings=(first, duplicate))


def test_hf_descriptor_requires_catalog_complete_logical_mapping():
    with pytest.raises(HFCompatibilityError, match="exactly cover"):
        _descriptor(parameter_mappings=(_mappings()[0],))


def test_hf_descriptor_rejects_catalog_extra_logical_mapping():
    extra = HFParameterMapping(
        "extra.weight", "extra/weight", "extra.weight", (1,), "float32"
    )
    with pytest.raises(HFCompatibilityError, match="exactly cover"):
        _descriptor(parameter_mappings=(*_mappings(), extra))


def test_hf_descriptor_rejects_catalog_shape_disagreement():
    first, second = _mappings()
    wrong = HFParameterMapping(
        second.logical_path,
        second.jax_pytree_path,
        second.hf_distribution_key,
        (8, 4),
        second.dtype,
    )
    with pytest.raises(HFCompatibilityError, match="shape"):
        _descriptor(parameter_mappings=(first, wrong))


def test_hf_descriptor_rejects_catalog_dtype_disagreement():
    first, second = _mappings()
    wrong = HFParameterMapping(
        second.logical_path,
        second.jax_pytree_path,
        second.hf_distribution_key,
        second.shape,
        "float16",
    )
    with pytest.raises(HFCompatibilityError, match="dtype"):
        _descriptor(parameter_mappings=(first, wrong))


def test_hf_descriptor_rejects_conflicting_architecture_configuration():
    with pytest.raises(HFCompatibilityError, match="conflicts"):
        _descriptor().validate_against(
            ArchitectureConfig("linear", model_config={"hidden_size": 8}, vocab_size=8),
            _catalog(),
        )


def test_hf_descriptor_rejects_conflicting_catalog_identity():
    other = ParameterCatalog("other", _catalog().parameters)
    with pytest.raises(HFCompatibilityError, match="catalog"):
        _descriptor().validate_against(_config(), other)


def test_hf_mapping_rejects_empty_or_zero_shape():
    with pytest.raises(HFCompatibilityError, match="positive"):
        HFParameterMapping("head.weight", "head/weight", "head.weight", (), "float32")
    with pytest.raises(HFCompatibilityError, match="positive"):
        HFParameterMapping("head.weight", "head/weight", "head.weight", (0,), "float32")


def test_hf_mapping_rejects_empty_tied_weight_group():
    with pytest.raises(HFCompatibilityError, match="tied_weight_group"):
        HFParameterMapping(
            "head.weight", "head/weight", "head.weight", (1,), "float32", ""
        )


@pytest.mark.parametrize(
    "name",
    [
        "mesh/device/weight",
        "fused_kernel.weight",
        "shard_0",
        "buffer_layout.weight",
        "process_index.weight",
    ],
)
def test_hf_mapping_rejects_runtime_dependent_names(name: str):
    with pytest.raises(HFCompatibilityError, match="runtime-layout"):
        HFParameterMapping(name, "jax/path", "hf.weight", (1,), "float32")


def test_hf_mapping_does_not_reject_unrelated_substrings():
    mapping = HFParameterMapping(
        "meshwork.weight", "head/weight", "classifier.weight", (1,), "float32"
    )
    assert mapping.logical_path == "meshwork.weight"


def test_hf_metadata_is_recursively_immutable_and_json_safe():
    mapping = HFParameterMapping(
        "head.weight",
        "head/weight",
        "classifier.weight",
        (1,),
        "float32",
        metadata={"nested": {"ids": [1, 2]}},
    )
    assert mapping.metadata["nested"]["ids"] == (1, 2)
    with pytest.raises(TypeError):
        mapping.metadata["new"] = 1


def test_hf_metadata_rejects_nonfinite_values():
    with pytest.raises(HFCompatibilityError, match="finite"):
        HFParameterMapping(
            "head.weight",
            "head/weight",
            "classifier.weight",
            (1,),
            "float32",
            metadata={"nan": float("nan")},
        )


def test_hf_special_tokens_must_be_distinct_and_in_vocabulary():
    with pytest.raises(HFCompatibilityError, match="duplicate"):
        _descriptor(special_token_ids={"pad": 0, "eos": 0})
    with pytest.raises(HFCompatibilityError, match="vocabulary"):
        _descriptor(special_token_ids={"pad": 8})


def test_hf_descriptor_mapping_axes_are_independent_contracts():
    mapping = _mappings()[1]
    assert (
        mapping.logical_path,
        mapping.jax_pytree_path,
        mapping.hf_distribution_key,
    ) == (
        "head.weight",
        "head/weight",
        "classifier.weight",
    )


def test_hf_descriptor_has_no_optional_runtime_dependencies():
    source = __import__("radjax_student.hf.contracts", fromlist=["__file__"]).__file__
    tree = ast.parse(open(source, encoding="utf-8").read())
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        for alias in (node.names if isinstance(node, ast.Import) else ())
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert {"transformers", "safetensors", "jax"}.isdisjoint(imports)
