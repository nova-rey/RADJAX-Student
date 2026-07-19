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
SPARSE_CROSS_ENTROPY_IDENTITY = ObjectiveIdentity(
    "radjax.objective.sparse_cross_entropy", "1"
)
SPARSE_CROSS_ENTROPY_METRIC_SCHEMA_ID = (
    "radjax.objective.sparse_cross_entropy.metrics.v1"
)


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


class SparseCategoricalCrossEntropyObjective:
    """Architecture-neutral mean token NLL for integer token-logit targets."""

    objective_id = SPARSE_CROSS_ENTROPY_IDENTITY.objective_id
    objective_version = SPARSE_CROSS_ENTROPY_IDENTITY.objective_version

    def objective_identity(self) -> ObjectiveIdentity:
        return SPARSE_CROSS_ENTROPY_IDENTITY

    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return ObjectiveCapabilityProfile(
            identity=self.objective_identity(),
            supported_execution_capabilities=("objective.jax_execution_v1",),
            required_surface_roles=("logits",),
            target_requirements=("targets.token_ids",),
            metric_schema_id=SPARSE_CROSS_ENTROPY_METRIC_SCHEMA_ID,
            metric_names=("objective.sparse_cross_entropy", "objective.token_accuracy"),
            non_claims=(
                "no_ignore_index",
                "no_label_smoothing",
                "no_masking",
                "no_token_weighting",
            ),
        )

    def execution_contract_version(self) -> str:
        return "objective.jax_execution.v1"

    def validate_config(self, config: ObjectiveConfig) -> None:
        if (
            not isinstance(config, ObjectiveConfig)
            or config.identity != self.objective_identity()
            or dict(config.values) != {"reduction": "mean"}
        ):
            raise ObjectiveContractError(
                "objective_config_invalid",
                "sparse cross-entropy requires canonical reduction=mean config",
            )

    def validate_resolved_surface(self, selection: ResolvedObjectiveSelection) -> None:
        if (
            not isinstance(selection, ResolvedObjectiveSelection)
            or selection.surface_role != "logits"
        ):
            raise ObjectiveContractError(
                "objective_surface_identity_mismatch",
                "sparse cross-entropy requires an architecture-owned logits surface",
            )

    def validate_targets(self, targets: Any) -> None:
        if not isinstance(targets, Mapping) or set(targets) != {"token_ids"}:
            raise ObjectiveContractError(
                "objective_target_invalid",
                "sparse cross-entropy requires only targets.token_ids",
            )
        target = targets["token_ids"]
        jnp = import_module("jax.numpy")
        if getattr(target, "ndim", None) != 2 or not jnp.issubdtype(
            target.dtype, jnp.integer
        ):
            raise ObjectiveContractError(
                "objective_target_invalid",
                "sparse cross-entropy targets.token_ids must be rank-2 integers",
            )
        jax = import_module("jax")
        if not isinstance(target, jax.core.Tracer) and bool(jnp.any(target < 0)):
            raise ObjectiveContractError(
                "objective_target_invalid",
                "sparse cross-entropy token IDs must be nonnegative",
            )

    def validate_metrics(self, metrics: Mapping[str, Any]) -> None:
        if tuple(sorted(metrics)) != self.capability_profile().metric_names:
            raise ObjectiveContractError(
                "objective_metric_invalid",
                "sparse cross-entropy emitted unknown or missing metrics",
            )
        for name, value in metrics.items():
            scalar = float(value)
            if scalar != scalar or scalar in (float("inf"), float("-inf")):
                raise ObjectiveContractError(
                    "objective_metric_invalid", f"metric {name} must be finite"
                )

    def evaluate_jax(
        self,
        *,
        surface: Any,
        targets: Any,
        weights: Any,
        config: ObjectiveConfig,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Return mean NLL and token accuracy without weighting or masking."""

        self.validate_config(config)
        if weights is not None and (not isinstance(weights, Mapping) or weights):
            raise ObjectiveContractError(
                "objective_target_invalid",
                "sparse cross-entropy does not support token weights or masks",
            )
        jax = import_module("jax")
        jnp = import_module("jax.numpy")
        target = targets["token_ids"]
        if (
            getattr(surface, "ndim", None) != 3
            or not jnp.issubdtype(surface.dtype, jnp.floating)
            or tuple(surface.shape[:2]) != tuple(target.shape)
            or surface.shape[-1] < 1
        ):
            raise ObjectiveContractError(
                "objective_target_invalid",
                "sparse cross-entropy requires logits [B,T,V] and targets [B,T]",
            )
        vocabulary_size = surface.shape[-1]
        invalid = (target < 0) | (target >= vocabulary_size)
        if not isinstance(target, jax.core.Tracer) and bool(jnp.any(invalid)):
            raise ObjectiveContractError(
                "objective_target_invalid",
                "sparse cross-entropy token IDs must be within the logits vocabulary",
            )
        safe_target = jnp.clip(target, 0, vocabulary_size - 1)
        log_probs = jax.nn.log_softmax(surface, axis=-1)
        nll = -jnp.take_along_axis(log_probs, safe_target[..., None], axis=-1)[..., 0]
        accuracy = jnp.mean(jnp.argmax(surface, axis=-1) == safe_target)
        invalid_value = jnp.any(invalid)
        loss = jnp.where(invalid_value, jnp.asarray(jnp.nan), jnp.mean(nll))
        accuracy = jnp.where(invalid_value, jnp.asarray(jnp.nan), accuracy)
        return loss, {
            "objective.sparse_cross_entropy": loss,
            "objective.token_accuracy": accuracy,
        }


__all__ = [
    "CANONICAL_MSE_IDENTITY",
    "MSE_METRIC_SCHEMA_ID",
    "MeanSquaredErrorObjective",
    "SPARSE_CROSS_ENTROPY_IDENTITY",
    "SPARSE_CROSS_ENTROPY_METRIC_SCHEMA_ID",
    "SparseCategoricalCrossEntropyObjective",
]
