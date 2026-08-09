"""P6.3 proves equation-authorized RWKV language and structural shapes."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from radjax_student.architecture import ArchitectureInitRequest
from radjax_student.architecture.rwkv7_reference import (
    RWKV7ReferencePlugin,
    configurable_architecture_config,
    parameter_catalog,
    reference_architecture_config,
)
from radjax_student.checkpoints import save_learning_checkpoint_v3
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    LearningBatch,
    ObjectiveConfig,
    ObjectiveScope,
    UpdateScope,
    hf_digest,
)
from radjax_student.learning import (
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningState,
    assemble_jax_learning_lifecycle,
)
from radjax_student.objectives import (
    SPARSE_CROSS_ENTROPY_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry, SgdOptimizer
from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry


def _config(vocabulary_size: int, context_length: int, **shape: int):
    return configurable_architecture_config(
        vocabulary_size,
        context_length,
        tokenizer=HFTokenizerIdentity(
            "p6-3-tokenizer",
            "v1",
            hf_digest({"vocabulary_size": vocabulary_size}),
            hf_digest({"context_length": context_length}),
            "synthetic",
            hf_digest({"normalization": "identity"}),
            "synthetic",
        ),
        vocabulary=HFVocabularyIdentity(
            vocabulary_size,
            hf_digest({"vocabulary_size": vocabulary_size}),
            hf_digest({"token_ids": vocabulary_size}),
            hf_digest({"added": []}),
            None,
        ),
        special_tokens=HFSpecialTokenIdentity(0, 1, 2, 3, None),
        **shape,
    )


def _forward(plugin, result, config, tokens):
    batch = SimpleNamespace(inputs={"token_ids": tokens})
    return plugin.apply_jax(
        result.parameters,
        result.architecture_carry,
        batch,
        architecture_config=config,
        objective_scope=ObjectiveScope(),
        training=False,
        rng_key=None,
    ).outputs


@pytest.mark.parametrize(
    "vocabulary_size, context_length", [(512, 8), (2048, 32), (4096, 64)]
)
def test_larger_vocabulary_context_rungs_are_config_derived(
    vocabulary_size: int, context_length: int
) -> None:
    plugin = RWKV7ReferencePlugin()
    config = _config(vocabulary_size, context_length)
    result = plugin.initialize_parameters(
        ArchitectureInitRequest(
            config=config,
            runtime_keys_reference="p6-3-language-rung",
            precision_policy="float32",
            runtime_initialization_material=jax.random.key(17),
        )
    )
    catalog = parameter_catalog(config)
    assert catalog.get("emb.weight").shape == (vocabulary_size, 8)
    assert catalog.get("head.weight").shape == (vocabulary_size, 8)
    tokens = jnp.arange(min(context_length, 8), dtype=jnp.int32)[None, :]
    eager = _forward(plugin, result, config, tokens)
    compiled = jax.jit(lambda values: _forward(plugin, result, config, values))(tokens)
    assert eager.shape == (1, min(context_length, 8), vocabulary_size)
    np.testing.assert_allclose(eager, compiled, rtol=1e-5, atol=1e-5)


def test_equation_authorized_structural_rung_maps_catalog_carry_and_kernel() -> None:
    plugin = RWKV7ReferencePlugin()
    config = _config(
        2048,
        32,
        hidden_size=16,
        layer_count=3,
        head_count=4,
        head_size=4,
        ffn_width=32,
    )
    result = plugin.initialize_parameters(
        ArchitectureInitRequest(
            config=config,
            runtime_keys_reference="p6-3-structural-rung",
            precision_policy="float32",
            runtime_initialization_material=jax.random.key(19),
        )
    )
    catalog = parameter_catalog(config)
    assert catalog.get("blocks.2.att.r_k").shape == (4, 4)
    assert catalog.get("blocks.2.att.w1").shape == (16, 32)
    assert catalog.get("blocks.2.att.w2").shape == (32, 16)
    assert catalog.get("blocks.2.ffn.key.weight").shape == (32, 16)
    assert catalog.get("blocks.2.ffn.value.weight").shape == (16, 32)
    assert result.architecture_carry["last_x_time"].shape == (3, 16)
    assert result.architecture_carry["time_state_matrix"].shape == (3, 4, 4, 4)
    tokens = jnp.arange(8, dtype=jnp.int32)[None, :]
    eager = _forward(plugin, result, config, tokens)
    compiled = jax.jit(lambda values: _forward(plugin, result, config, values))(tokens)
    assert eager.shape == (1, 8, 2048)
    np.testing.assert_allclose(eager, compiled, rtol=1e-5, atol=1e-5)


def test_structural_divisibility_constraint_fails_closed() -> None:
    with pytest.raises(ValueError, match="hidden_size must equal"):
        _config(512, 8, hidden_size=10, head_count=3, head_size=4)


def test_frozen_phase4_config_remains_unchanged() -> None:
    plugin = RWKV7ReferencePlugin()
    frozen = reference_architecture_config()
    result = plugin.initialize_parameters(
        ArchitectureInitRequest(
            config=frozen,
            runtime_keys_reference="p6-3-frozen",
            precision_policy="float32",
            runtime_initialization_material=jax.random.key(23),
        )
    )
    assert frozen == reference_architecture_config()
    assert result.parameter_catalog == plugin.describe_parameters(
        architecture_config=frozen
    )
    assert result.architecture_carry["time_state_matrix"].shape == (2, 2, 4, 4)


def test_structural_shape_reaches_generic_lifecycle_and_checkpoint_restore(
    tmp_path,
) -> None:
    from radjax_student.architecture import ArchitectureRegistry
    from radjax_student.architecture.rwkv7_reference import register_rwkv7_reference

    architectures = ArchitectureRegistry()
    register_rwkv7_reference(architectures)
    optimizers = OptimizerRegistry()
    optimizers.register(SgdOptimizer())
    registries = JaxLearningAssemblyRegistries(
        architectures,
        build_default_objective_registry(),
        optimizers,
        build_default_runtime_registry(),
    )
    config = _config(
        2048,
        32,
        hidden_size=16,
        layer_count=3,
        head_count=4,
        head_size=4,
        ffn_width=32,
    )
    assembled = assemble_jax_learning_lifecycle(
        JaxLearningAssemblyRequest(
            architecture_id="radjax.architecture.rwkv7_reference",
            architecture_version=1,
            architecture_config=config,
            objective_identity=SPARSE_CROSS_ENTROPY_IDENTITY,
            objective_config=ObjectiveConfig(
                SPARSE_CROSS_ENTROPY_IDENTITY, {"reduction": "mean"}
            ),
            optimizer_id="sgd.v1",
            optimizer_version=1,
            optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.01),
            runtime_backend_id="jax",
            runtime_implementation_version="p2.9",
            runtime_config=RuntimeConfig(
                backend_id="jax",
                platform_preference="cpu",
                precision_policy="float32",
                placement_policy="single_device",
                compilation_policy="eager",
                distributed_policy="disabled",
                fallback_policy="disallowed",
                seed=29,
            ),
            root_seed=29,
            learning_state=LearningState(
                "p6-3-structural-lifecycle",
                active_update_scope=UpdateScope(),
                active_objective_scope=ObjectiveScope(),
            ),
        ),
        registries=registries,
    )
    lifecycle = assembled.loop_executor.lifecycle
    batch = LearningBatch(
        "p6-3-structural-lifecycle",
        inputs={"token_ids": [list(range(8))]},
        targets={"token_ids": [list(range(1, 8)) + [0]]},
    )
    execution = assembled.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        objective=lifecycle.objective_selection,
        batch=batch,
    )
    assert execution.runtime_result.status == "pass"
    assert execution.result.changed_parameter_paths
    checkpoint = tmp_path / "structural-checkpoint"
    saved = save_learning_checkpoint_v3(
        lifecycle.checkpoint(), checkpoint, optimizer=lifecycle.optimizer
    )
    restored = lifecycle.restore_from_checkpoint(checkpoint)
    assert saved.integrity["algorithm"] == "sha256"
    assert restored.config_digest == lifecycle.config_digest
