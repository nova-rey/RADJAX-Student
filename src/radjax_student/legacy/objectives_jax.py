"""Deprecated historical JAX objective compatibility, never production authority.

The builders in this module retain pre-P3.12A mathematics for compatibility
evidence only. Production execution must resolve a complete objective plugin
through :mod:`radjax_student.objectives` before it reaches the JAX step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol
from warnings import warn

from radjax_student.architecture import (
    ForwardResult,
    JaxArchitectureExecution,
    JaxArchitecturePlugin,
)
from radjax_student.contracts import ObjectiveScope, ResolvedObjectiveSelection
from radjax_student.learning.jax_core import JaxBatch, JaxLossAuxiliary


def _warn() -> None:
    warn(
        "radjax_student.legacy.objectives_jax is deprecated and non-production; "
        "resolve an ObjectiveRegistry selection for modern execution",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class LegacyJaxObjectiveConfig:
    """Pre-P3.12A split objective configuration retained only for old evidence."""

    objective_id: str
    objective_scope: ObjectiveScope = ObjectiveScope()
    reduction: Literal["mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        _warn()
        if not self.objective_id or not isinstance(
            self.objective_scope, ObjectiveScope
        ):
            raise ValueError("JAX objective identifiers must be nonempty")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("unsupported JAX objective reduction")

    @property
    def surface_id(self) -> str:
        if self.objective_scope.kind in {"final_output", "whole_student"}:
            return "final_output"
        assert self.objective_scope.target_id is not None
        return self.objective_scope.target_id

    def tree_flatten(self):
        return (), (
            self.objective_id,
            self.objective_scope.kind,
            self.objective_scope.target_id,
            self.reduction,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        objective_id, kind, target_id, reduction = aux_data
        return cls(
            objective_id,
            ObjectiveScope(kind=kind, target_id=target_id),
            reduction,
        )


_JAX_CONFIG_REGISTERED = False


def _jax():
    """Load JAX only after an explicit deprecated compatibility invocation."""

    global _JAX_CONFIG_REGISTERED
    jax = import_module("jax")
    if not _JAX_CONFIG_REGISTERED:
        jax.tree_util.register_pytree_node_class(LegacyJaxObjectiveConfig)
        _JAX_CONFIG_REGISTERED = True
    return jax


class LegacyJaxObjective(Protocol):
    """Historical evaluate-only protocol. It cannot enter modern execution."""

    def evaluate(
        self,
        surface: Any,
        targets: Any,
        weights: Any,
        objective_config: LegacyJaxObjectiveConfig,
    ) -> tuple[Any, Mapping[str, Any]]: ...


def build_legacy_jax_loss_fn(
    architecture: JaxArchitectureExecution,
    objective: LegacyJaxObjective,
):
    """Build the deprecated pre-registry loss graph for compatibility tests."""

    _warn()
    jax = _jax()
    if not isinstance(architecture, JaxArchitectureExecution):
        raise TypeError("architecture must expose the JAX execution capability")

    def loss_fn(
        parameters: Any,
        architecture_carry: Any,
        batch: JaxBatch,
        objective_config: LegacyJaxObjectiveConfig,
        rng_key: Any | None,
    ):
        if not isinstance(batch, JaxBatch):
            raise TypeError("batch must be JaxBatch")
        input_carry = jax.tree_util.tree_map(jax.lax.stop_gradient, architecture_carry)
        forward = architecture.apply_jax(
            parameters,
            input_carry,
            batch,
            objective_scope=objective_config.objective_scope,
            training=True,
            rng_key=rng_key,
        )
        if not isinstance(forward, ForwardResult):
            raise TypeError("JAX architecture execution must return ForwardResult")
        surface = forward.surface_for_legacy_scope(objective_config.objective_scope)
        loss, objective_auxiliary = objective.evaluate(
            surface,
            batch.targets,
            batch.weights,
            objective_config,
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


def build_legacy_resolved_jax_loss_fn(
    architecture: JaxArchitecturePlugin,
    objective: LegacyJaxObjective,
    objective_selection: ResolvedObjectiveSelection,
):
    """Build the deprecated architecture-resolved historical loss graph."""

    _warn()
    jax = _jax()
    if not isinstance(architecture, JaxArchitecturePlugin):
        raise TypeError("resolved JAX execution requires JaxArchitecturePlugin")
    if not isinstance(objective_selection, ResolvedObjectiveSelection):
        raise TypeError("objective_selection must be ResolvedObjectiveSelection")

    def loss_fn(
        parameters: Any,
        architecture_carry: Any,
        batch: JaxBatch,
        objective_config: LegacyJaxObjectiveConfig,
        rng_key: Any | None,
    ):
        if not isinstance(batch, JaxBatch):
            raise TypeError("batch must be JaxBatch")
        input_carry = jax.tree_util.tree_map(jax.lax.stop_gradient, architecture_carry)
        forward = architecture.apply_jax(
            parameters,
            input_carry,
            batch,
            objective_scope=objective_selection.scope,
            training=True,
            rng_key=rng_key,
        )
        if not isinstance(forward, ForwardResult):
            raise TypeError("JAX architecture execution must return ForwardResult")
        surface = forward.surface_for(objective_selection)
        loss, objective_auxiliary = objective.evaluate(
            surface,
            batch.targets,
            batch.weights,
            objective_config,
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


__all__ = [
    "LegacyJaxObjective",
    "LegacyJaxObjectiveConfig",
    "build_legacy_jax_loss_fn",
    "build_legacy_resolved_jax_loss_fn",
]
