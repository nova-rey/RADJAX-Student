"""Independent P5.0A proof for stateless configurable RWKV-7 instantiation."""

from __future__ import annotations

import pytest

from radjax_student.architecture import ArchitectureConfig, ArchitectureContractError
from radjax_student.architecture.models import ArchitectureInitRequest
from radjax_student.architecture.rwkv7_reference import (
    RWKV7ReferencePlugin,
    configurable_architecture_config,
    parameter_catalog,
    parameter_layout,
)
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    LearningBatch,
    ObjectiveScope,
    hf_digest,
)

pytestmark = pytest.mark.jax


def _config() -> ArchitectureConfig:
    return configurable_architecture_config(
        64,
        5,
        tokenizer=HFTokenizerIdentity(
            "synthetic-rwkv7-tokenizer",
            "v1",
            hf_digest({"tokenizer": 64}),
            hf_digest({"config": 64}),
            "synthetic",
            hf_digest({"normalization": "none"}),
            "synthetic",
        ),
        vocabulary=HFVocabularyIdentity(
            64,
            hf_digest({"vocabulary": 64}),
            hf_digest({"token_to_id": 64}),
            hf_digest({"added": []}),
            None,
        ),
        special_tokens=HFSpecialTokenIdentity(0, 1, 2, 3, None),
    )


def test_configurable_64_by_5_initializes_and_executes_without_plugin_state() -> None:
    import jax
    import jax.numpy as jnp

    from radjax_student.learning.jax_core import JaxBatch

    config = _config()
    plugin = RWKV7ReferencePlugin()
    assert set(plugin.__dict__) == {"architecture_id", "architecture_version"}
    assert parameter_catalog(config).get("emb.weight").shape == (64, 8)
    assert parameter_catalog(config).get("head.weight").shape == (64, 8)
    assert parameter_layout(config).entries == parameter_layout(config).entries

    request = ArchitectureInitRequest(
        config=config,
        runtime_keys_reference="p5_0a.synthetic:17",
        precision_policy="float32",
        runtime_initialization_material=jax.random.key(17),
    )
    result = plugin.initialize_parameters(request)
    repeated = plugin.initialize_parameters(request)
    assert result.parameter_layout == parameter_layout(config)
    assert result.hf_descriptor is not None
    assert (
        result.hf_descriptor.tokenizer.to_dict()
        == config.metadata["hf_language_identities"]["tokenizer"]
    )
    assert (
        result.hf_descriptor.vocabulary.to_dict()
        == config.metadata["hf_language_identities"]["vocabulary"]
    )
    assert result.hf_descriptor.special_tokens == HFSpecialTokenIdentity.from_dict(
        config.metadata["hf_language_identities"]["special_tokens"]
    )
    assert jax.numpy.array_equal(
        result.parameters["emb"]["weight"], repeated.parameters["emb"]["weight"]
    )

    for length in (1, 4, 5):
        batch = LearningBatch(
            batch_id=f"p5_0a-length-{length}",
            inputs={"token_ids": [[0] * length]},
            targets={},
        )
        assert plugin.validate_batch(batch, config).status == "pass"
    for invalid in ([[0] * 6], [[64]]):
        assert (
            plugin.validate_batch(
                LearningBatch(
                    batch_id="p5_0a-invalid", inputs={"token_ids": invalid}, targets={}
                ),
                config,
            ).status
            == "fail"
        )

    for length in (1, 4, 5):
        forward = plugin.apply_jax(
            result.parameters,
            result.architecture_carry,
            JaxBatch(
                inputs={"token_ids": jnp.arange(length, dtype=jnp.int32)[None, :]},
                targets={},
            ),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        )
        assert forward.outputs.shape == (1, length, 64)

    for token_id in (0, 63):
        assert plugin.apply_jax(
            result.parameters,
            result.architecture_carry,
            JaxBatch(inputs={"token_ids": jnp.asarray([[token_id]])}, targets={}),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        ).outputs.shape == (1, 1, 64)

    with pytest.raises(ArchitectureContractError) as caught:
        plugin.apply_jax(
            result.parameters,
            result.architecture_carry,
            JaxBatch(
                inputs={"token_ids": jnp.asarray([[0, 1, 2, 3, 4, 5]])},
                targets={},
            ),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        )
    assert caught.value.code == "architecture_batch_incompatible"

    with pytest.raises(ArchitectureContractError) as caught:
        plugin.apply_jax(
            result.parameters,
            result.architecture_carry,
            JaxBatch(inputs={"token_ids": jnp.asarray([[64]])}, targets={}),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        )
    assert caught.value.code == "architecture_batch_incompatible"


def test_configurable_requires_complete_neutral_language_identity() -> None:
    config = _config()
    payload = config.to_dict()
    payload["metadata"] = {"hf_language_identities": {"tokenizer": "opaque"}}
    with pytest.raises(ArchitectureContractError, match="complete valid neutral"):
        RWKV7ReferencePlugin().validate_config(ArchitectureConfig.from_dict(payload))

    with pytest.raises(TypeError, match="HFTokenizerIdentity"):
        configurable_architecture_config(
            64,
            5,
            tokenizer="opaque-tokenizer-hash",  # type: ignore[arg-type]
            vocabulary=HFVocabularyIdentity(
                64, hf_digest(1), hf_digest(2), hf_digest(3), None
            ),
            special_tokens=HFSpecialTokenIdentity(None, None, None, None, None),
        )
    with pytest.raises(ValueError, match="match"):
        configurable_architecture_config(
            64,
            5,
            tokenizer=HFTokenizerIdentity(
                "synthetic", "v1", hf_digest(1), hf_digest(2), "synthetic", hf_digest(3)
            ),
            vocabulary=HFVocabularyIdentity(
                63, hf_digest(1), hf_digest(2), hf_digest(3), None
            ),
            special_tokens=HFSpecialTokenIdentity(None, None, None, None, None),
        )
    with pytest.raises(ValueError, match="positive"):
        configurable_architecture_config(
            0,
            5,
            tokenizer=HFTokenizerIdentity(
                "synthetic", "v1", hf_digest(1), hf_digest(2), "synthetic", hf_digest(3)
            ),
            vocabulary=HFVocabularyIdentity(
                1, hf_digest(1), hf_digest(2), hf_digest(3), None
            ),
            special_tokens=HFSpecialTokenIdentity(None, None, None, None, None),
        )
