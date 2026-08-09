"""Registered neutral behavioral objectives for the generic JAX lifecycle."""

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

BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY = ObjectiveIdentity(
    "radjax.objective.behavioral_corridor", "1"
)
BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY = ObjectiveIdentity(
    "radjax.objective.behavioral_exemplar", "1"
)
BEHAVIORAL_CORRIDOR_METRIC_SCHEMA_ID = "radjax.objective.behavioral_corridor.metrics.v1"
BEHAVIORAL_EXEMPLAR_METRIC_SCHEMA_ID = "radjax.objective.behavioral_exemplar.metrics.v1"
BEHAVIORAL_OBJECTIVE_CONFIG = {
    "policy_id": "radjax.student.behavior_objective.v1",
    "reduction": "assignment_attention_weighted_mean.v1",
}
_STATISTICS = ("entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass")
_CORRIDOR_METRICS = tuple(
    sorted(
        (
            *(f"corridor.{name}.inside_rate" for name in _STATISTICS),
            *(f"corridor.{name}.mean_squared_distance" for name in _STATISTICS),
            "corridor.all_statistics.inside_rate",
            "corridor.loss",
        )
    )
)


class _BehavioralObjectiveBase:
    objective_id: str
    objective_version: str

    def objective_identity(self) -> ObjectiveIdentity:
        return ObjectiveIdentity(self.objective_id, self.objective_version)

    def execution_contract_version(self) -> str:
        return "objective.jax_execution.v1"

    def validate_config(self, config: ObjectiveConfig) -> None:
        if (
            not isinstance(config, ObjectiveConfig)
            or config.identity != self.objective_identity()
            or dict(config.values) != BEHAVIORAL_OBJECTIVE_CONFIG
        ):
            raise ObjectiveContractError(
                "objective_config_invalid",
                "behavioral objective requires the frozen neutral policy",
            )

    def validate_resolved_surface(self, selection: ResolvedObjectiveSelection) -> None:
        if (
            not isinstance(selection, ResolvedObjectiveSelection)
            or selection.surface_role != "logits"
        ):
            raise ObjectiveContractError(
                "objective_surface_identity_mismatch",
                "behavioral objective requires architecture-owned logits",
            )

    def validate_metrics(self, metrics: Mapping[str, Any]) -> None:
        if tuple(sorted(metrics)) != self.capability_profile().metric_names:
            raise ObjectiveContractError(
                "objective_metric_invalid",
                "behavioral objective metrics are incomplete",
            )
        for value in metrics.values():
            scalar = float(value)
            if scalar != scalar or scalar in (float("inf"), float("-inf")):
                raise ObjectiveContractError(
                    "objective_metric_invalid",
                    "behavioral objective metric is nonfinite",
                )


class BehavioralCorridorObjective(_BehavioralObjectiveBase):
    """P5.5 corridor interval loss over neutral B=1 logits and targets."""

    objective_id = BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY.objective_id
    objective_version = BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY.objective_version

    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return ObjectiveCapabilityProfile(
            identity=self.objective_identity(),
            supported_execution_capabilities=("objective.jax_execution_v1",),
            required_surface_roles=("logits",),
            target_requirements=(
                "targets.position",
                "targets.assignment_weight",
                "targets.statistic_lower",
                "targets.statistic_upper",
                "weights.attention_mask",
            ),
            metric_schema_id=BEHAVIORAL_CORRIDOR_METRIC_SCHEMA_ID,
            metric_names=_CORRIDOR_METRICS,
            non_claims=(
                "no_artifact_access",
                "no_checkpoint_access",
                "no_architecture_identity",
            ),
        )

    def validate_targets(self, targets: Any) -> None:
        _validate_corridor_targets(targets)

    def evaluate_jax(
        self, *, surface: Any, targets: Any, weights: Any, config: ObjectiveConfig
    ) -> tuple[Any, Mapping[str, Any]]:
        self.validate_config(config)
        _validate_corridor_targets(targets)
        if not isinstance(weights, Mapping) or set(weights) != {"attention_mask"}:
            raise ObjectiveContractError(
                "objective_target_invalid", "corridor objective requires attention mask"
            )
        return behavioral_corridor_loss_v1(
            logits=surface,
            positions=targets["position"],
            assignment_weights=targets["assignment_weight"],
            attention_mask=weights["attention_mask"],
            lower_bounds=targets["statistic_lower"],
            upper_bounds=targets["statistic_upper"],
        )


class BehavioralExemplarObjective(_BehavioralObjectiveBase):
    """P5.5 exemplar coarse cross entropy over neutral B=1 logits and targets."""

    objective_id = BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY.objective_id
    objective_version = BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY.objective_version

    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return ObjectiveCapabilityProfile(
            identity=self.objective_identity(),
            supported_execution_capabilities=("objective.jax_execution_v1",),
            required_surface_roles=("logits",),
            target_requirements=(
                "targets.selected_position",
                "targets.top_token_ids",
                "targets.top_probabilities",
                "targets.aggregate_tail_mass",
            ),
            metric_schema_id=BEHAVIORAL_EXEMPLAR_METRIC_SCHEMA_ID,
            metric_names=("exemplar.coarse_cross_entropy",),
            non_claims=(
                "no_artifact_access",
                "no_checkpoint_access",
                "no_architecture_identity",
            ),
        )

    def validate_targets(self, targets: Any) -> None:
        _validate_exemplar_targets(targets)

    def evaluate_jax(
        self, *, surface: Any, targets: Any, weights: Any, config: ObjectiveConfig
    ) -> tuple[Any, Mapping[str, Any]]:
        self.validate_config(config)
        _validate_exemplar_targets(targets)
        if weights is not None and (not isinstance(weights, Mapping) or weights):
            raise ObjectiveContractError(
                "objective_target_invalid", "exemplar objective does not accept weights"
            )
        return behavioral_exemplar_loss_v1(
            logits=surface,
            selected_positions=targets["selected_position"],
            top_token_ids=targets["top_token_ids"],
            top_probabilities=targets["top_probabilities"],
            aggregate_tail_mass=targets["aggregate_tail_mass"],
        )


def behavioral_corridor_loss_v1(
    *,
    logits: Any,
    positions: Any,
    assignment_weights: Any,
    attention_mask: Any,
    lower_bounds: Any,
    upper_bounds: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """P5.5 interval loss expressed only in neutral tensors."""

    jax = import_module("jax")
    jnp = import_module("jax.numpy")
    logits = jnp.asarray(logits)
    _validate_logits(logits)
    batch_size, sequence_length = logits.shape[:2]
    _validate_vector(positions, batch_size, "corridor positions", integer=True)
    _validate_vector(assignment_weights, batch_size, "corridor assignment weights")
    if getattr(attention_mask, "shape", None) != (batch_size, sequence_length):
        raise ObjectiveContractError(
            "objective_target_invalid", "corridor attention mask shape is invalid"
        )
    if getattr(lower_bounds, "shape", None) != (
        batch_size,
        len(_STATISTICS),
    ) or getattr(upper_bounds, "shape", None) != (batch_size, len(_STATISTICS)):
        raise ObjectiveContractError(
            "objective_target_invalid", "corridor statistic bounds shape is invalid"
        )
    rows = jnp.arange(batch_size, dtype=jnp.int32)
    safe_positions = jnp.clip(positions, 0, sequence_length - 1)
    selected_logits = logits[rows, safe_positions]
    probabilities = jax.nn.softmax(selected_logits, axis=-1)
    log_probabilities = jax.nn.log_softmax(selected_logits, axis=-1)
    ordered = jnp.sort(probabilities, axis=-1)[:, ::-1]

    def top(count: int) -> Any:
        return jnp.sum(ordered[:, : min(count, logits.shape[-1])], axis=-1)

    statistics = (
        -jnp.sum(probabilities * log_probabilities, axis=-1),
        ordered[:, 0] - ordered[:, 1],
        top(8),
        top(32),
        1.0 - top(32),
    )
    mask = jnp.asarray(attention_mask)[rows, safe_positions].astype(logits.dtype)
    weights = jnp.asarray(assignment_weights, dtype=logits.dtype) * mask
    denominator = jnp.sum(weights)
    outside_position = (positions < 0) | (positions >= sequence_length)
    invalid = (denominator <= 0) | jnp.any(outside_position)
    metrics: dict[str, Any] = {}
    loss = jnp.asarray(0.0, dtype=logits.dtype)
    all_inside = jnp.ones((batch_size,), dtype=bool)
    for index, name in enumerate(_STATISTICS):
        statistic = statistics[index]
        outside = jnp.maximum(lower_bounds[:, index] - statistic, 0.0) + jnp.maximum(
            statistic - upper_bounds[:, index], 0.0
        )
        squared = outside**2
        inside = squared == 0.0
        all_inside = jnp.logical_and(all_inside, inside)
        metrics[f"corridor.{name}.inside_rate"] = (
            jnp.sum(weights * inside) / denominator
        )
        metrics[f"corridor.{name}.mean_squared_distance"] = (
            jnp.sum(weights * squared) / denominator
        )
        loss = loss + jnp.sum(weights * squared) / denominator
    metrics["corridor.all_statistics.inside_rate"] = (
        jnp.sum(weights * all_inside) / denominator
    )
    loss = jnp.where(invalid, jnp.asarray(jnp.nan), loss)
    metrics = {
        key: jnp.where(invalid, jnp.asarray(jnp.nan), value)
        for key, value in metrics.items()
    }
    metrics["corridor.loss"] = loss
    return loss, metrics


def behavioral_exemplar_loss_v1(
    *,
    logits: Any,
    selected_positions: Any,
    top_token_ids: Any,
    top_probabilities: Any,
    aggregate_tail_mass: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """P5.5 coarse CE expressed only in neutral tensors."""

    jax = import_module("jax")
    jnp = import_module("jax.numpy")
    logits = jnp.asarray(logits)
    _validate_logits(logits)
    batch_size, sequence_length, vocabulary_size = logits.shape
    _validate_vector(
        selected_positions, batch_size, "exemplar selected positions", integer=True
    )
    if (
        getattr(top_token_ids, "ndim", None) != 2
        or top_token_ids.shape[0] != batch_size
        or top_token_ids.shape[1] < 1
        or getattr(top_probabilities, "shape", None) != top_token_ids.shape
        or getattr(aggregate_tail_mass, "shape", None) != (batch_size,)
    ):
        raise ObjectiveContractError(
            "objective_target_invalid", "exemplar sparse targets are invalid"
        )
    rows = jnp.arange(batch_size, dtype=jnp.int32)
    safe_positions = jnp.clip(selected_positions, 0, sequence_length - 1)
    token_logits = logits[rows, safe_positions]
    invalid_tokens = (top_token_ids < 0) | (top_token_ids >= vocabulary_size)
    safe_ids = jnp.clip(top_token_ids, 0, vocabulary_size - 1)
    log_probs = jax.nn.log_softmax(token_logits, axis=-1)
    active_log_probs = jnp.take_along_axis(log_probs, safe_ids, axis=1)
    complement = (
        jnp.ones_like(token_logits, dtype=bool).at[rows[:, None], safe_ids].set(False)
    )
    tail_log_probability = jax.scipy.special.logsumexp(
        jnp.where(complement, log_probs, -jnp.inf), axis=-1
    )
    active_terms = jnp.where(
        top_probabilities > 0, -top_probabilities * active_log_probs, 0.0
    )
    per_row = jnp.sum(active_terms, axis=-1) + jnp.where(
        aggregate_tail_mass > 0,
        -aggregate_tail_mass * tail_log_probability,
        0.0,
    )
    invalid = (
        jnp.any((selected_positions < 0) | (selected_positions >= sequence_length))
        | jnp.any(invalid_tokens)
        | jnp.any(top_probabilities < 0)
        | jnp.any(aggregate_tail_mass < 0)
        | jnp.any(
            jnp.abs(jnp.sum(top_probabilities, axis=-1) + aggregate_tail_mass - 1.0)
            > 1e-6
        )
    )
    loss = jnp.where(invalid, jnp.asarray(jnp.nan), jnp.mean(per_row))
    return loss, {"exemplar.coarse_cross_entropy": loss}


def _validate_corridor_targets(targets: Any) -> None:
    if not isinstance(targets, Mapping) or set(targets) != {
        "position",
        "mode_id",
        "assignment_weight",
        "statistic_lower",
        "statistic_upper",
    }:
        raise ObjectiveContractError(
            "objective_target_invalid", "corridor targets are incomplete"
        )


def _validate_exemplar_targets(targets: Any) -> None:
    if not isinstance(targets, Mapping) or set(targets) != {
        "selected_position",
        "top_token_ids",
        "top_probabilities",
        "aggregate_tail_mass",
    }:
        raise ObjectiveContractError(
            "objective_target_invalid", "exemplar targets are incomplete"
        )


def _validate_logits(logits: Any) -> None:
    if (
        getattr(logits, "ndim", None) != 3
        or logits.shape[0] < 1
        or logits.shape[1] < 1
        or logits.shape[2] < 2
    ):
        raise ObjectiveContractError(
            "objective_target_invalid", "behavioral objective requires logits [B,T,V]"
        )


def _validate_vector(
    value: Any, size: int, name: str, *, integer: bool = False
) -> None:
    jnp = import_module("jax.numpy")
    if getattr(value, "shape", None) != (size,) or (
        integer and not jnp.issubdtype(value.dtype, jnp.integer)
    ):
        raise ObjectiveContractError("objective_target_invalid", f"{name} are invalid")


__all__ = [
    "BEHAVIORAL_CORRIDOR_METRIC_SCHEMA_ID",
    "BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY",
    "BEHAVIORAL_EXEMPLAR_METRIC_SCHEMA_ID",
    "BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY",
    "BEHAVIORAL_OBJECTIVE_CONFIG",
    "BehavioralCorridorObjective",
    "BehavioralExemplarObjective",
    "behavioral_corridor_loss_v1",
    "behavioral_exemplar_loss_v1",
]
