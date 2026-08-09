"""Explicit construction for built-in objective registry contents."""

from __future__ import annotations

from radjax_student.objectives.behavioral import (
    BehavioralCorridorObjective,
    BehavioralExemplarObjective,
)
from radjax_student.objectives.jax import (
    MeanSquaredErrorObjective,
    SparseCategoricalCrossEntropyObjective,
)
from radjax_student.objectives.registry import ObjectiveRegistry


def build_default_objective_registry() -> ObjectiveRegistry:
    registry = ObjectiveRegistry()
    registry.register(MeanSquaredErrorObjective())
    registry.register(SparseCategoricalCrossEntropyObjective())
    registry.register(BehavioralCorridorObjective())
    registry.register(BehavioralExemplarObjective())
    return registry


__all__ = ["build_default_objective_registry"]
