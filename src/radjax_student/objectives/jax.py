"""JAX implementation for the canonical mean-squared-error objective."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from radjax_student.contracts import (
    ObjectiveCapabilityProfile,
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveIdentity,
    ResolvedObjectiveSelection,
)

CANONICAL_MSE_IDENTITY = ObjectiveIdentity("radjax.objective.mean_squared_error", "1")
MSE_METRIC_SCHEMA_ID = "radjax.objective.mean_squared_error.metrics.v1"


class MeanSquaredErrorObjective:
    """Canonical MSE objective. JAX is loaded only when it executes."""

    objective_id = CANONICAL_MSE_IDENTITY.objective_id
    objective_version = CANONICAL_MSE_IDENTITY.objective_version

    def objective_identity(self) -> ObjectiveIdentity:
        return CANONICAL_MSE_IDENTITY

    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return ObjectiveCapabilityProfile(
            identity=self.objective_identity(),
            supported_execution_capabilities=("objective.jax_execution_v1",),
            required_surface_roles=("prediction",),
            target_requirements=("targets.y",),
            metric_schema_id=MSE_METRIC_SCHEMA_ID,
            metric_names=("objective.mse",),
            non_claims=(
                "no_tome_objective",
                "no_distillation",
                "no_model_quality_claim",
            ),
        )

    def execution_contract_version(self) -> str:
        return "objective.jax_execution.v1"

    def validate_config(self, config: ObjectiveConfig) -> None:
        if (
            not isinstance(config, ObjectiveConfig)
            or config.identity != self.objective_identity()
        ):
            raise ObjectiveContractError(
                "objective_config_identity_mismatch",
                "MSE config identity does not match canonical MSE objective",
            )
        if set(config.values) != {"reduction"} or config.values["reduction"] != "mean":
            raise ObjectiveContractError(
                "objective_config_invalid",
                "MSE requires canonical reduction=mean config",
            )

    def validate_resolved_surface(self, selection: ResolvedObjectiveSelection) -> None:
        if (
            not isinstance(selection, ResolvedObjectiveSelection)
            or selection.surface_role != "prediction"
        ):
            raise ObjectiveContractError(
                "objective_surface_identity_mismatch",
                "MSE requires an architecture-owned prediction surface",
            )

    def validate_targets(self, targets: Any) -> None:
        if not isinstance(targets, Mapping) or "y" not in targets:
            raise ObjectiveContractError(
                "objective_target_invalid", "MSE requires targets.y"
            )
        # This executes at the public step boundary before graph construction.
        # Keeping it here avoids converting a traced JAX value to Python inside
        # the compiled loss graph while still rejecting non-finite user input.
        jnp = import_module("jax.numpy")
        if not bool(jnp.all(jnp.isfinite(targets["y"]))):
            raise ObjectiveContractError(
                "objective_target_invalid", "MSE targets.y must be finite"
            )

    def validate_metrics(self, metrics: Mapping[str, Any]) -> None:
        profile = self.capability_profile()
        if tuple(sorted(metrics)) != profile.metric_names:
            raise ObjectiveContractError(
                "objective_metric_invalid",
                "MSE emitted unknown or missing objective metrics",
            )
        for name, value in metrics.items():
            scalar = float(value)
            if scalar != scalar or scalar in (float("inf"), float("-inf")):
                raise ObjectiveContractError(
                    "objective_metric_invalid", f"MSE metric {name} must be finite"
                )

    def evaluate_jax(
        self,
        *,
        surface: Any,
        targets: Any,
        weights: Any,
        config: ObjectiveConfig,
    ) -> tuple[Any, Mapping[str, Any]]:
        del weights
        self.validate_config(config)
        jnp = import_module("jax.numpy")
        target = targets["y"]
        if getattr(surface, "shape", None) != getattr(target, "shape", None):
            raise ObjectiveContractError(
                "objective_target_invalid",
                "MSE prediction and targets.y shapes must match",
            )
        loss = jnp.mean((surface - target) ** 2)
        return loss, {"objective.mse": loss}


__all__ = [
    "CANONICAL_MSE_IDENTITY",
    "MSE_METRIC_SCHEMA_ID",
    "MeanSquaredErrorObjective",
]
