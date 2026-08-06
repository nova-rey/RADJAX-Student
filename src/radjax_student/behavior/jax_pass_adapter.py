"""Narrow lazy-JAX execution seam for architecture-neutral behavior passes.

The adapter owns only the JAX mechanics needed by the pre-generic-lifecycle
corridor, exemplar, and held-out paths.  It has no artifact, checkpoint,
optimizer, or architecture-plugin knowledge.  P6.2 may replace these callers
with the genuine registered lifecycle; until then this preserves the accepted
P5 surface while keeping optional-runtime imports out of behavior policies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

BEHAVIOR_JAX_PASS_ADAPTER_V1 = "radjax.student.behavior_jax_pass_adapter.v1"


@dataclass(frozen=True)
class BehaviorJaxPassComputationV1:
    """The exact numerical output required by one neutral behavior pass."""

    loss: Any
    gradients: Any
    gradient_norm: float


def materialize_behavior_jax_inputs_v1(
    input_ids: Any, attention_mask: Any
) -> tuple[Any, Any]:
    """Convert the two neutral input values at the optional-runtime boundary."""

    jnp = import_module("jax.numpy")
    return jnp.asarray(input_ids), jnp.asarray(attention_mask)


def execute_behavior_jax_pass_v1(
    *,
    parameters: Any,
    input_ids: Any,
    attention_mask: Any,
    forward: Callable[[Any, Any, Any], Any],
    objective: Callable[[Any], Any],
) -> BehaviorJaxPassComputationV1:
    """Differentiate one neutral forward/objective pair without policy knowledge."""

    jax = import_module("jax")
    jnp = import_module("jax.numpy")
    jax_input_ids, jax_attention_mask = materialize_behavior_jax_inputs_v1(
        input_ids, attention_mask
    )

    def loss_fn(candidate: Any) -> Any:
        return objective(forward(candidate, jax_input_ids, jax_attention_mask))

    loss, gradients = jax.value_and_grad(loss_fn)(parameters)
    leaves = jax.tree_util.tree_leaves(gradients)
    gradient_norm = float(jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves)))
    return BehaviorJaxPassComputationV1(loss, gradients, gradient_norm)


def jax_tree_leaves_v1(value: Any) -> list[Any]:
    """Expose tree leaves needed to bind changed optimizer paths."""

    return list(import_module("jax").tree_util.tree_leaves(value))


def jax_tree_digest_payload_v1(value: Any) -> dict[str, object]:
    """Return canonical host values for an already-authorized checkpoint digest."""

    leaves, structure = import_module("jax").tree_util.tree_flatten(value)
    return {
        "structure": str(structure),
        "leaves": [np.asarray(leaf).tolist() for leaf in leaves],
    }
