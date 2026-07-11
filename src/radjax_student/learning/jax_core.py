"""Optional pure-JAX learning composition for the Phase 3.5 boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from radjax_student.architecture import ForwardResult, JaxArchitectureExecution


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxBatch:
    """Runtime value batch; the finite-JSON LearningBatch remains separate."""

    inputs: Any
    targets: Any
    weights: Any = None

    def tree_flatten(self):
        return (self.inputs, self.targets, self.weights), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxObjectiveConfig:
    objective_id: str
    surface_id: str = "final_output"
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if not self.objective_id or not self.surface_id:
            raise ValueError("JAX objective identifiers must be nonempty")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("unsupported JAX objective reduction")

    def tree_flatten(self):
        return (), (self.objective_id, self.surface_id, self.reduction)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


class JaxObjective(Protocol):
    def evaluate(
        self,
        surface: Any,
        targets: Any,
        weights: Any,
        objective_config: JaxObjectiveConfig,
    ) -> tuple[Any, Mapping[str, Any]]: ...


def build_jax_loss_fn(
    architecture: JaxArchitectureExecution,
    objective: JaxObjective,
):
    """Build a pure loss function; compilation belongs to runtime."""

    if not isinstance(architecture, JaxArchitectureExecution):
        raise TypeError("architecture must expose the JAX execution capability")

    def loss_fn(
        parameters: Any,
        architecture_state: Any,
        batch: JaxBatch,
        objective_config: JaxObjectiveConfig,
        rng_key: Any | None,
    ):
        if not isinstance(batch, JaxBatch):
            raise TypeError("batch must be JaxBatch")
        state = jax.tree_util.tree_map(jax.lax.stop_gradient, architecture_state)
        forward = architecture.apply_jax(
            parameters,
            state,
            batch,
            objective_scope=objective_config.surface_id,
            training=True,
            rng_key=rng_key,
        )
        if not isinstance(forward, ForwardResult):
            raise TypeError("JAX architecture execution must return ForwardResult")
        surface = forward.surface(objective_config.surface_id)
        loss, objective_auxiliary = objective.evaluate(
            surface,
            batch.targets,
            batch.weights,
            objective_config,
        )
        auxiliary = {
            "selected_surface": surface,
            "updated_architecture_state": (
                architecture_state
                if forward.updated_architecture_state is None
                else forward.updated_architecture_state
            ),
            "updated_runtime_state": (
                architecture_state
                if forward.updated_runtime_state is None
                else forward.updated_runtime_state
            ),
            "objective": dict(objective_auxiliary),
        }
        return loss, auxiliary

    return loss_fn


def build_value_and_grad_fn(loss_fn):
    """Create the autodiff function without selecting an execution mode."""
    return jax.value_and_grad(loss_fn, argnums=0, has_aux=True)


def validate_finite_loss_and_gradients(loss: Any, gradients: Any) -> None:
    if not bool(jnp.all(jnp.isfinite(loss))):
        raise ValueError("JAX loss must be finite")
    leaves = jax.tree_util.tree_leaves(gradients)
    if not leaves or not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
        raise ValueError("JAX gradients must be finite")


def apply_scoped_gradient_update(
    parameters: Any,
    gradients: Any,
    selection_mask: Any,
    learning_rate: float,
) -> Any:
    """Apply a pure pytree update using a same-structure boolean mask."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    parameter_tree = jax.tree_util.tree_structure(parameters)
    if parameter_tree != jax.tree_util.tree_structure(gradients):
        raise ValueError("parameters and gradients must share a pytree structure")
    if parameter_tree != jax.tree_util.tree_structure(selection_mask):
        raise ValueError("selection mask must share the parameter pytree structure")
    return jax.tree_util.tree_map(
        lambda parameter, gradient, selected: (
            parameter - learning_rate * gradient if selected else parameter
        ),
        parameters,
        gradients,
        selection_mask,
    )
