"""Optional pure-JAX learning composition for the Phase 3.5 boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from radjax_student.architecture import (
    ForwardResult,
    JaxArchitecturePlugin,
)
from radjax_student.contracts import (
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveExecutionDescriptor,
    ResolvedObjectiveSelection,
)
from radjax_student.objectives import JaxObjectivePlugin, ObjectiveRegistrySelection


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxBatch:
    """Runtime value batch; the finite-JSON LearningBatch remains separate."""

    inputs: Any
    targets: Any
    weights: Any = None
    source_batch_digest: str | None = None

    def tree_flatten(self):
        return (self.inputs, self.targets, self.weights), self.source_batch_digest

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, source_batch_digest=aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxLossAuxiliary:
    """Typed functional output of one JAX architecture/objective evaluation."""

    selected_surface: Any
    updated_architecture_carry: Any
    objective_metrics: Mapping[str, Any]
    architecture_metrics: Mapping[str, Any]

    def tree_flatten(self):
        return (
            self.selected_surface,
            self.updated_architecture_carry,
            dict(self.objective_metrics),
            dict(self.architecture_metrics),
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def build_registered_jax_loss_fn(
    *,
    architecture: JaxArchitecturePlugin,
    objective_selection: ObjectiveRegistrySelection,
    objective_config: ObjectiveConfig,
    objective_descriptor: ObjectiveExecutionDescriptor,
    resolved_selection: ResolvedObjectiveSelection,
):
    """Build the sole production objective loss graph from registry selection.

    This path deliberately has no parameter-tree argument on the objective
    boundary. The architecture consumes parameters; the objective only receives
    its declared forward surface and validated targets.
    """

    if not isinstance(architecture, JaxArchitecturePlugin):
        raise TypeError("registered JAX loss requires JaxArchitecturePlugin")
    if not isinstance(objective_selection, ObjectiveRegistrySelection):
        raise TypeError("registered JAX loss requires ObjectiveRegistrySelection")
    if not objective_selection.is_registry_selected:
        raise ObjectiveContractError(
            "objective_identity_mismatch",
            "registered JAX loss requires an ObjectiveRegistry selection",
        )
    if not isinstance(objective_selection.plugin, JaxObjectivePlugin):
        raise TypeError("selected objective lacks complete JAX execution capability")
    if not isinstance(objective_config, ObjectiveConfig):
        raise TypeError("registered JAX loss requires ObjectiveConfig")
    if not isinstance(objective_descriptor, ObjectiveExecutionDescriptor):
        raise TypeError("registered JAX loss requires ObjectiveExecutionDescriptor")
    if not isinstance(resolved_selection, ResolvedObjectiveSelection):
        raise TypeError("registered JAX loss requires ResolvedObjectiveSelection")
    plugin = objective_selection.plugin
    if (
        objective_config.identity != objective_selection.identity
        or objective_descriptor.identity != objective_selection.identity
        or objective_descriptor.capability_profile_digest
        != objective_selection.profile.digest
        or objective_descriptor.config_digest != objective_config.digest
        or objective_descriptor.resolved_surface_identity != resolved_selection.digest
        or objective_descriptor.metric_schema_id
        != objective_selection.profile.metric_schema_id
        or objective_descriptor.implementation_identity
        != objective_selection.implementation_identity
    ):
        raise ObjectiveContractError(
            "objective_config_identity_mismatch",
            "registered objective descriptor is internally inconsistent",
        )
    plugin.validate_config(objective_config)
    plugin.validate_resolved_surface(resolved_selection)

    def loss_fn(
        parameters: Any,
        architecture_carry: Any,
        batch: JaxBatch,
        rng_key: Any | None,
    ):
        if not isinstance(batch, JaxBatch):
            raise TypeError("batch must be JaxBatch")
        input_carry = jax.tree_util.tree_map(jax.lax.stop_gradient, architecture_carry)
        forward = architecture.apply_jax(
            parameters,
            input_carry,
            batch,
            objective_scope=resolved_selection.scope,
            training=True,
            rng_key=rng_key,
        )
        if not isinstance(forward, ForwardResult):
            raise TypeError("JAX architecture execution must return ForwardResult")
        surface = forward.surface_for(resolved_selection)
        loss, objective_auxiliary = plugin.evaluate_jax(
            surface=surface,
            targets=batch.targets,
            weights=batch.weights,
            config=objective_config,
        )
        next_carry = (
            input_carry
            if forward.updated_architecture_carry is None
            else forward.updated_architecture_carry
        )
        next_carry = jax.tree_util.tree_map(jax.lax.stop_gradient, next_carry)
        return loss, JaxLossAuxiliary(
            selected_surface=surface,
            updated_architecture_carry=next_carry,
            objective_metrics=dict(objective_auxiliary),
            architecture_metrics=dict(forward.architecture_metrics),
        )

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


__all__ = [
    "JaxBatch",
    "JaxLossAuxiliary",
    "build_registered_jax_loss_fn",
    "build_value_and_grad_fn",
    "validate_finite_loss_and_gradients",
]
