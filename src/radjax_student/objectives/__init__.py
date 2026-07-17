"""Production objective identity, registry, and optional JAX execution boundary."""

from radjax_student.objectives.protocols import (
    JaxObjectiveExecution,
    JaxObjectivePlugin,
    ObjectivePlugin,
)
from radjax_student.objectives.registry import (
    ObjectiveRegistry,
    ObjectiveRegistrySelection,
    implementation_identity_for,
)

__all__ = [
    "JaxObjectiveExecution",
    "JaxObjectivePlugin",
    "ObjectivePlugin",
    "ObjectiveRegistry",
    "ObjectiveRegistrySelection",
    "implementation_identity_for",
    "build_default_objective_registry",
    "CANONICAL_MSE_IDENTITY",
    "MSE_METRIC_SCHEMA_ID",
    "MeanSquaredErrorObjective",
    "CanonicalObjectiveAliasSelection",
    "HISTORICAL_MSE_ALIASES",
    "resolve_historical_objective_alias",
]


def __getattr__(name: str):
    if name == "build_default_objective_registry":
        from radjax_student.objectives.builtin import build_default_objective_registry

        return build_default_objective_registry
    if name in {
        "CanonicalObjectiveAliasSelection",
        "HISTORICAL_MSE_ALIASES",
        "resolve_historical_objective_alias",
    }:
        from radjax_student.objectives.legacy import (
            HISTORICAL_MSE_ALIASES,
            CanonicalObjectiveAliasSelection,
            resolve_historical_objective_alias,
        )

        return {
            "CanonicalObjectiveAliasSelection": CanonicalObjectiveAliasSelection,
            "HISTORICAL_MSE_ALIASES": HISTORICAL_MSE_ALIASES,
            "resolve_historical_objective_alias": resolve_historical_objective_alias,
        }[name]
    if name in {
        "CANONICAL_MSE_IDENTITY",
        "MSE_METRIC_SCHEMA_ID",
        "MeanSquaredErrorObjective",
    }:
        from radjax_student.objectives.jax import (
            CANONICAL_MSE_IDENTITY,
            MSE_METRIC_SCHEMA_ID,
            MeanSquaredErrorObjective,
        )

        return {
            "CANONICAL_MSE_IDENTITY": CANONICAL_MSE_IDENTITY,
            "MSE_METRIC_SCHEMA_ID": MSE_METRIC_SCHEMA_ID,
            "MeanSquaredErrorObjective": MeanSquaredErrorObjective,
        }[name]
    raise AttributeError(name)
