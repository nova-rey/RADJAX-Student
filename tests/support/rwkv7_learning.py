"""P4 test-only assembly helpers for the explicit RWKV learning fixture."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.architecture.rwkv7_reference import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    reference_architecture_config,
    register_rwkv7_reference,
)
from radjax_student.contracts import ObjectiveConfig, ObjectiveScope, UpdateScope
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

TOKENS = (1, 7, 3, 5)
TARGETS = (7, 3, 5, 1)


def batch(tokens: tuple[int, ...] = TOKENS) -> LearningBatch:
    """Return the one finite-JSON sequence accepted by the RWKV reference."""

    return LearningBatch(
        "p45-rwkv7",
        inputs={"token_ids": [list(tokens)]},
        targets={"token_ids": [list(TARGETS)]},
    )


def assembled(compilation_policy: str = "eager"):
    """Build the real registry-selected P3.12C/P3.12D RWKV lifecycle."""

    architecture_registry = ArchitectureRegistry()
    register_rwkv7_reference(architecture_registry)
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(SgdOptimizer())
    request = JaxLearningAssemblyRequest(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        architecture_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        architecture_config=reference_architecture_config(),
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
            "p45-rwkv7",
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
    """Run one step through the lifecycle's generic loop executor."""

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
    """Compare two JAX pytrees at the declared float32 proof tolerance."""

    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    return first_tree == second_tree and all(
        bool(jnp.allclose(a, b, rtol=1e-5, atol=2e-5))
        for a, b in zip(first_leaves, second_leaves, strict=True)
    )


def tree_changed(first: Any, second: Any) -> bool:
    """Return whether any corresponding JAX leaf differs exactly."""

    return any(
        not bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(first),
            jax.tree_util.tree_leaves(second),
            strict=True,
        )
    )


def all_finite(value: Any) -> bool:
    """Return whether every JAX leaf is finite."""

    return all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(value)
    )


__all__ = [
    "TARGETS",
    "TOKENS",
    "all_finite",
    "assembled",
    "batch",
    "execute",
    "tree_allclose",
    "tree_changed",
]
