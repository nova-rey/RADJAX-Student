"""Explicit migration boundary for pre-P3.12 objective aliases.

This module translates historical names into the canonical registry selection.
It never accepts a callable or wraps an arbitrary objective implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from radjax_student.contracts import (
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveExecutionDescriptor,
    ResolvedObjectiveSelection,
)
from radjax_student.objectives.jax import CANONICAL_MSE_IDENTITY
from radjax_student.objectives.registry import (
    ObjectiveRegistry,
    ObjectiveRegistrySelection,
)

HISTORICAL_MSE_ALIASES = frozenset(
    {
        "mse",
        "linear.mse.v1",
        "stateful_linear_mse.v1",
        CANONICAL_MSE_IDENTITY.objective_id,
    }
)


@dataclass(frozen=True)
class CanonicalObjectiveAliasSelection:
    """Canonical registry selection obtained from one historical alias."""

    source_alias: str
    selection: ObjectiveRegistrySelection
    config: ObjectiveConfig
    descriptor: ObjectiveExecutionDescriptor


def resolve_historical_objective_alias(
    *,
    source_alias: str,
    registry: ObjectiveRegistry,
    resolved_selection: ResolvedObjectiveSelection,
) -> CanonicalObjectiveAliasSelection:
    """Resolve an accepted name before lifecycle construction.

    The architecture owns ``resolved_selection``. This boundary owns only name
    translation and registry selection; callers cannot provide an implementation.
    """

    if source_alias not in HISTORICAL_MSE_ALIASES:
        raise ObjectiveContractError(
            "objective_identity_mismatch",
            "historical objective alias is not supported",
            details={"source_alias": source_alias},
        )
    selection = registry.select(CANONICAL_MSE_IDENTITY)
    config = ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"})
    descriptor = registry.execution_descriptor(
        selection=selection,
        config=config,
        resolved_selection=resolved_selection,
    )
    return CanonicalObjectiveAliasSelection(
        source_alias=source_alias,
        selection=selection,
        config=config,
        descriptor=descriptor,
    )


__all__ = [
    "CanonicalObjectiveAliasSelection",
    "HISTORICAL_MSE_ALIASES",
    "resolve_historical_objective_alias",
]
