"""Focused generic-lifecycle helpers for the configurable Mamba-2 fixture."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.architecture.mamba2_reference import (
    MAMBA2_REFERENCE_ARCHITECTURE_ID,
    MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
    configurable_architecture_config,
    register_mamba2_reference,
)
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    ObjectiveConfig,
    ObjectiveScope,
    UpdateScope,
    hf_digest,
)
from radjax_student.learning import (
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningBatch,
    LearningState,
    assemble_jax_learning_lifecycle,
)
from radjax_student.objectives import (
    SPARSE_CROSS_ENTROPY_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry, SgdOptimizer
from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry

VOCABULARY_SIZE = 512
CONTEXT_LENGTH = 8
TOKENS = (1, 7, 3, 5)
TARGETS = (7, 3, 5, 1)


def config():
    return configurable_architecture_config(
        VOCABULARY_SIZE,
        CONTEXT_LENGTH,
        tokenizer=HFTokenizerIdentity(
            "synthetic-mamba2-tokenizer",
            "v1",
            hf_digest({"tokenizer": VOCABULARY_SIZE}),
            hf_digest({"config": CONTEXT_LENGTH}),
            "synthetic",
            hf_digest({"normalization": "none"}),
            "synthetic",
        ),
        vocabulary=HFVocabularyIdentity(
            VOCABULARY_SIZE,
            hf_digest({"vocabulary": VOCABULARY_SIZE}),
            hf_digest({"token_to_id": VOCABULARY_SIZE}),
            hf_digest({"added": []}),
            None,
        ),
        special_tokens=HFSpecialTokenIdentity(0, 1, 2, 3, None),
    )


def batch(tokens: tuple[int, ...] = TOKENS) -> LearningBatch:
    return LearningBatch(
        "m2-v512-t8",
        inputs={"token_ids": [list(tokens)]},
        targets={"token_ids": [list(TARGETS)]},
    )


def assembled(compilation_policy: str = "eager"):
    architecture_registry = ArchitectureRegistry()
    register_mamba2_reference(architecture_registry)
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(SgdOptimizer())
    request = JaxLearningAssemblyRequest(
        architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
        architecture_version=MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
        architecture_config=config(),
        objective_identity=SPARSE_CROSS_ENTROPY_IDENTITY,
        objective_config=ObjectiveConfig(
            SPARSE_CROSS_ENTROPY_IDENTITY, {"reduction": "mean"}
        ),
        optimizer_id="sgd.v1",
        optimizer_version=1,
        optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.05),
        runtime_backend_id="jax",
        runtime_implementation_version="p2.9",
        runtime_config=RuntimeConfig(
            backend_id="jax",
            platform_preference="cpu",
            precision_policy="float32",
            placement_policy="single_device",
            compilation_policy=compilation_policy,
            distributed_policy="disabled",
            fallback_policy="disallowed",
            seed=17,
        ),
        root_seed=17,
        learning_state=LearningState(
            "m2-v512-t8",
            active_update_scope=UpdateScope("whole_student"),
            active_objective_scope=ObjectiveScope(),
        ),
    )
    return assemble_jax_learning_lifecycle(
        request,
        registries=JaxLearningAssemblyRegistries(
            architecture_registry,
            build_default_objective_registry(),
            optimizer_registry,
            build_default_runtime_registry(),
        ),
    )


def execute(assembled_lifecycle, learning_batch: LearningBatch):
    lifecycle = assembled_lifecycle.loop_executor.lifecycle
    return assembled_lifecycle.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        parameters=lifecycle.parameters,
        objective=lifecycle.objective_selection,
        batch=learning_batch,
    )


def tree_allclose(first: Any, second: Any) -> bool:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    return first_tree == second_tree and all(
        bool(jnp.allclose(a, b, rtol=1e-5, atol=2e-5))
        for a, b in zip(first_leaves, second_leaves, strict=True)
    )


def tree_changed(first: Any, second: Any) -> bool:
    return any(
        not bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(first),
            jax.tree_util.tree_leaves(second),
            strict=True,
        )
    )


def all_finite(value: Any) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(value)
    )


__all__ = [
    "CONTEXT_LENGTH",
    "TARGETS",
    "TOKENS",
    "VOCABULARY_SIZE",
    "all_finite",
    "assembled",
    "batch",
    "config",
    "execute",
    "tree_allclose",
    "tree_changed",
]
