from __future__ import annotations

import json

import pytest

from radjax_student.architecture import ArchitectureConfig
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


def _mapping() -> HFParameterMapping:
    return HFParameterMapping(
        "head.weight", "head/weight", "classifier.weight", (4, 8), "float32"
    )


def test_hf_descriptor_round_trip_preserves_identity_and_unknown_fields():
    descriptor = HFCompatibilityDescriptor.from_architecture(
        _config(),
        model_type="radjax-linear",
        tokenizer_id="toy-tokenizer",
        special_token_ids={"pad": 0, "eos": 2},
        parameter_mappings=(_mapping(),),
        architecture_state_metadata={"state_schema": "v1"},
    )
    payload = {**descriptor.to_dict(), "future_field": {"enabled": True}}
    restored = HFCompatibilityDescriptor.from_dict(payload)
    assert restored.to_dict()["future_field"] == {"enabled": True}
    assert restored.to_json() == json.dumps(
        restored.to_dict(), sort_keys=True, separators=(",", ":")
    )
    restored.validate_against(_config())


def test_hf_descriptor_rejects_conflicting_architecture_configuration():
    descriptor = HFCompatibilityDescriptor.from_architecture(
        _config(),
        model_type="radjax-linear",
        tokenizer_id="toy-tokenizer",
        special_token_ids={"pad": 0},
        parameter_mappings=(_mapping(),),
    )
    with pytest.raises(HFCompatibilityError, match="conflicts"):
        descriptor.validate_against(
            ArchitectureConfig(
                architecture_id="linear",
                model_config={"hidden_size": 8},
                vocab_size=8,
            )
        )


@pytest.mark.parametrize("name", ["mesh/device/weight", "fused.weight", "shard_0"])
def test_hf_mapping_rejects_runtime_dependent_names(name: str):
    with pytest.raises(HFCompatibilityError, match="runtime-layout"):
        HFParameterMapping(name, "jax/path", "hf.weight", (1,), "float32")
