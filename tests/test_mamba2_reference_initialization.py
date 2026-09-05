"""M2.3 deterministic initialization and layout evidence."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from radjax_student.architecture import (
    ArchitectureContractError,
    ArchitectureInitRequest,
)
from radjax_student.architecture.mamba2_reference import (
    Mamba2ReferencePlugin,
    configurable_architecture_config,
    reference_architecture_config,
)
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
)
from radjax_student.runtime.jax_bridge import materialize_initialization_jax_key


def _request(config, reference: str) -> ArchitectureInitRequest:
    return ArchitectureInitRequest(
        config=config,
        runtime_keys_reference=reference,
        precision_policy="float32",
        runtime_initialization_material=materialize_initialization_jax_key(reference),
    )


def _identity(size: int):
    return (
        HFTokenizerIdentity(
            "m2", "fixture", "a" * 64, "b" * 64, "synthetic", "c" * 64, "synthetic"
        ),
        HFVocabularyIdentity(size, "d" * 64, "e" * 64, "f" * 64, None),
        HFSpecialTokenIdentity(0, 1, 1, 2, None),
    )


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return tree


@pytest.mark.jax
def test_initialization_is_complete_deterministic_and_tied() -> None:
    plugin = Mamba2ReferencePlugin()
    config = reference_architecture_config()
    first = plugin.initialize_parameters(
        _request(config, "runtime_keys.v1:initialization:1")
    )
    second = plugin.initialize_parameters(
        _request(config, "runtime_keys.v1:initialization:1")
    )
    assert first.parameters is not None and second.parameters is not None
    assert first.parameter_layout is not None
    first.parameter_layout.validate_materialized_parameters(first.parameters)
    for entry in first.parameter_layout.entries:
        assert jnp.array_equal(
            _leaf(first.parameters, entry.jax_keypath),
            _leaf(second.parameters, entry.jax_keypath),
        )
    assert (
        first.parameters["backbone"]["embedding"]["weight"]
        is first.parameters["lm_head"]["weight"]
    )
    assert set(first.architecture_carry) == {
        "layers.0.conv_state",
        "layers.0.ssm_state",
        "layers.1.conv_state",
        "layers.1.ssm_state",
    }
    assert all(
        jnp.count_nonzero(value) == 0 for value in first.architecture_carry.values()
    )


@pytest.mark.jax
def test_configurable_v512_t8_initializes_from_same_plugin() -> None:
    tokenizer, vocabulary, special = _identity(512)
    config = configurable_architecture_config(
        512, 8, tokenizer=tokenizer, vocabulary=vocabulary, special_tokens=special
    )
    result = Mamba2ReferencePlugin().initialize_parameters(
        _request(config, "runtime_keys.v1:initialization:2")
    )
    assert result.parameter_layout is not None and result.parameters is not None
    result.parameter_layout.validate_materialized_parameters(result.parameters)
    assert result.parameters["backbone"]["embedding"]["weight"].shape == (512, 8)
    assert result.hf_descriptor is not None
    assert result.hf_descriptor.vocabulary.vocabulary_size == 512
    assert result.hf_reference == result.hf_descriptor.preservation_reference()


def test_initialization_rejects_missing_material_and_non_float32() -> None:
    plugin = Mamba2ReferencePlugin()
    config = reference_architecture_config()
    with pytest.raises(ArchitectureContractError) as missing:
        plugin.initialize_parameters(
            ArchitectureInitRequest(
                config, "runtime_keys.v1:initialization:3", "float32"
            )
        )
    assert missing.value.code == "architecture_initialization_failed"
    with pytest.raises(ArchitectureContractError) as precision:
        plugin.initialize_parameters(
            ArchitectureInitRequest(
                config,
                "runtime_keys.v1:initialization:3",
                "bfloat16",
                runtime_initialization_material=materialize_initialization_jax_key(
                    "runtime_keys.v1:initialization:3"
                ),
            )
        )
    assert precision.value.code == "architecture_initialization_failed"


def test_plugin_metadata_matches_initialization_capability_profile() -> None:
    plugin = Mamba2ReferencePlugin()
    assert (
        plugin.architecture_metadata().capability_profile == plugin.capability_profile()
    )
