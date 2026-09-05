"""M2.2 static Mamba-2 schema and registration checks."""

from __future__ import annotations

import json
import subprocess
import sys

from radjax_student.architecture.mamba2_reference import (
    MAMBA2_REFERENCE_ARCHITECTURE_ID,
    Mamba2ReferencePlugin,
    architecture_metadata,
    carry_descriptor,
    configurable_architecture_config,
    hf_descriptor,
    parameter_catalog,
    parameter_layout,
    reference_architecture_config,
    register_mamba2_reference,
)
from radjax_student.architecture.registry import ArchitectureRegistry
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    LearningBatch,
)


def _identity(vocabulary_size: int):
    tokenizer = HFTokenizerIdentity(
        "synthetic-mamba-tokenizer",
        "fixture",
        "a" * 64,
        "b" * 64,
        "synthetic",
        "c" * 64,
        "synthetic",
    )
    vocabulary = HFVocabularyIdentity(
        vocabulary_size, "d" * 64, "e" * 64, "f" * 64, None
    )
    special = HFSpecialTokenIdentity(0, 1, 1, 2, None)
    return tokenizer, vocabulary, special


def test_frozen_profile_is_static_and_json_finite() -> None:
    config = reference_architecture_config()
    assert config.vocab_size == 16
    assert config.sequence_length == 4
    assert config.model_config["d_model"] == 8
    assert config.model_config["dt_limit"]["max"] == "UNBOUNDED"
    encoded = json.dumps(config.to_dict(), allow_nan=False, sort_keys=True)
    assert "Infinity" not in encoded


def test_language_dimensions_are_configurable_without_structural_scaling() -> None:
    tokenizer, vocabulary, special = _identity(512)
    config = configurable_architecture_config(
        512,
        8,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        special_tokens=special,
    )
    assert config.vocab_size == 512
    assert config.sequence_length == 8
    assert config.model_config["d_model"] == 8
    assert parameter_layout(config).entry_for_logical_path(
        "backbone.embedding.weight"
    ).shape == (512, 8)
    descriptor = hf_descriptor(config)
    assert descriptor.vocabulary.vocabulary_size == 512
    assert (
        descriptor.special_tokens.eos_token_id == descriptor.special_tokens.pad_token_id
    )
    json.dumps(descriptor.to_dict(), allow_nan=False)


def test_catalog_and_layout_cover_tied_embedding_and_head() -> None:
    catalog = parameter_catalog()
    layout = parameter_layout()
    assert len(catalog.parameters) == 21
    assert set(layout.logical_paths) == set(catalog.paths)
    embedding = layout.entry_for_logical_path("backbone.embedding.weight")
    head = layout.entry_for_logical_path("lm_head.weight")
    assert embedding.tied_weight_group == head.tied_weight_group == "token_embedding"
    assert embedding.shape == head.shape == (16, 8)
    assert all(
        item.metadata["mapping_class"]
        in {"direct", "transformed but equation-preserving"}
        for item in catalog.parameters
    )


def test_carry_descriptor_has_separate_persistent_state_families() -> None:
    persistent = carry_descriptor()["persistent_leaves"]
    assert set(persistent) == {
        "layers.0.conv_state",
        "layers.0.ssm_state",
        "layers.1.conv_state",
        "layers.1.ssm_state",
    }
    assert persistent["layers.0.conv_state"]["shape"] == [1, 24, 4]
    assert persistent["layers.0.ssm_state"]["shape"] == [1, 4, 4, 4]


def test_registration_is_explicit_and_static_only() -> None:
    registry = ArchitectureRegistry()
    plugin = register_mamba2_reference(registry)
    assert isinstance(plugin, Mamba2ReferencePlugin)
    assert registry.list_plugins() == (MAMBA2_REFERENCE_ARCHITECTURE_ID,)
    profile = plugin.capability_profile()
    assert "architecture.parameter_initialization_v1" in profile.capabilities
    assert "architecture.jax_execution_v1" not in profile.capabilities
    assert plugin.capability_profile().metadata["phase"] == "M2.3"


def test_static_plugin_batch_validation_accepts_short_configured_chunks() -> None:
    plugin = Mamba2ReferencePlugin()
    tokenizer, vocabulary, special = _identity(512)
    config = configurable_architecture_config(
        512,
        8,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        special_tokens=special,
    )
    batch = LearningBatch("b", inputs={"token_ids": [[1, 7, 3]]})
    assert plugin.validate_batch(batch, config).status == "pass"
    too_long = LearningBatch("b2", inputs={"token_ids": [[1] * 9]})
    assert plugin.validate_batch(too_long, config).status == "fail"


def test_static_import_does_not_load_jax() -> None:
    code = (
        "import sys; import radjax_student.architecture.mamba2_reference; "
        "assert 'jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_metadata_is_config_derived() -> None:
    tokenizer, vocabulary, special = _identity(512)
    config = configurable_architecture_config(
        512,
        8,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        special_tokens=special,
    )
    metadata = architecture_metadata(config)
    assert metadata.parameter_catalog.get("backbone.embedding.weight").shape == (512, 8)
    assert metadata.region("block_1").parameter_paths
