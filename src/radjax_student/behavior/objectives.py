"""P5.5 behavioral objectives over the neutral P5.4 batch boundary.

These functions receive logits and immutable neutral batches only.  They do
not know about artifacts, manifests, locators, architecture parameters, or
producer delivery formats.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from radjax_student.behavior.models import CorridorBatchV1, ExemplarBatchV1

_STATISTICS = ("entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass")
BEHAVIOR_OBJECTIVE_POLICY_V1 = "radjax.student.behavior_objective.v1"
BEHAVIOR_OBJECTIVE_REDUCTION_V1 = "assignment_attention_weighted_mean.v1"


class BehavioralObjectiveError(ValueError):
    """Neutral behavioral values cannot define the requested objective."""


@dataclass(frozen=True)
class BehavioralObjectivePolicyV1:
    """Versioned Student-owned weights and normalization, not teacher data."""

    statistic_weights: Mapping[str, float]
    policy_id: str = BEHAVIOR_OBJECTIVE_POLICY_V1
    reduction_id: str = BEHAVIOR_OBJECTIVE_REDUCTION_V1

    def __post_init__(self) -> None:
        if self.policy_id != BEHAVIOR_OBJECTIVE_POLICY_V1:
            raise BehavioralObjectiveError("unsupported behavioral objective policy")
        if self.reduction_id != BEHAVIOR_OBJECTIVE_REDUCTION_V1:
            raise BehavioralObjectiveError("unsupported behavioral objective reduction")
        if set(self.statistic_weights) != set(_STATISTICS):
            raise BehavioralObjectiveError(
                "statistic weights must cover exactly P5.5 statistics"
            )
        if any(
            not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
            for value in self.statistic_weights.values()
        ):
            raise BehavioralObjectiveError(
                "statistic weights must be finite and positive"
            )


DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1 = BehavioralObjectivePolicyV1(
    statistic_weights={name: 1.0 for name in _STATISTICS}
)


def corridor_objective_v1(
    logits: Any,
    batch: CorridorBatchV1,
    *,
    policy: BehavioralObjectivePolicyV1 = DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1,
) -> tuple[Any, Mapping[str, Any]]:
    """Return inclusive-interval squared corridor loss and auditable metrics."""

    _validate_corridor_inputs(logits, batch, policy)
    jax = import_module("jax")
    jnp = import_module("jax.numpy")
    logits = jnp.asarray(logits)
    # P5.4 keeps Contract coordinates global. Map them to the local neutral
    # batch rows deterministically; no artifact coordinate or path is needed.
    global_rows = np.unique(batch.example_indices)
    local_rows = np.searchsorted(global_rows, batch.example_indices)
    rows = jnp.asarray(local_rows, dtype=jnp.int32)
    positions = jnp.asarray(batch.positions, dtype=jnp.int32)
    probabilities = jax.nn.softmax(logits[rows, positions], axis=-1)
    log_probabilities = jax.nn.log_softmax(logits[rows, positions], axis=-1)
    ordered = jnp.sort(probabilities, axis=-1)[:, ::-1]
    vocabulary_size = probabilities.shape[-1]

    def top(count: int) -> Any:
        return jnp.sum(ordered[:, : min(count, vocabulary_size)], axis=-1)

    statistics = {
        "entropy": -jnp.sum(probabilities * log_probabilities, axis=-1),
        "top1_margin": ordered[:, 0] - ordered[:, 1],
        "top8_mass": top(8),
        "top32_mass": top(32),
        "tail_mass": 1.0 - top(32),
    }
    lower, upper = _corridor_bounds(batch)
    mask = jnp.asarray(batch.attention_mask)[rows, positions].astype(logits.dtype)
    weights = jnp.asarray(batch.assignment_weights, dtype=logits.dtype) * mask
    denominator = jnp.sum(weights)
    metrics: dict[str, Any] = {}
    weighted_loss = jnp.asarray(0.0, dtype=logits.dtype)
    all_inside = jnp.ones_like(weights, dtype=bool)
    for index, name in enumerate(_STATISTICS):
        value = statistics[name]
        outside = jnp.maximum(lower[:, index] - value, 0.0) + jnp.maximum(
            value - upper[:, index], 0.0
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
        weighted_loss = weighted_loss + (
            jnp.asarray(policy.statistic_weights[name], dtype=logits.dtype)
            * jnp.sum(weights * squared)
            / denominator
        )
    metrics["corridor.all_statistics.inside_rate"] = (
        jnp.sum(weights * all_inside) / denominator
    )
    metrics["corridor.loss"] = weighted_loss
    return weighted_loss, metrics


def exemplar_coarse_cross_entropy_v1(
    logits: Any, batch: ExemplarBatchV1
) -> tuple[Any, Mapping[str, Any]]:
    """Cross entropy on singleton declared tokens plus one aggregate tail."""

    _validate_exemplar_inputs(logits, batch)
    jax = import_module("jax")
    jnp = import_module("jax.numpy")
    logits = jnp.asarray(logits)
    losses = []
    for row, target in enumerate(batch.sparse_targets):
        token_ids = jnp.asarray(target.token_ids, dtype=jnp.int32)
        token_logits = logits[row, int(batch.selected_positions[row])]
        log_probs = jax.nn.log_softmax(token_logits)
        active_log_probs = log_probs[token_ids]
        # The aggregate outcome is represented directly as the logsumexp of
        # all *unlisted* token outcomes; it never invents a teacher tail split.
        complement = (
            jnp.ones(token_logits.shape[0], dtype=bool).at[token_ids].set(False)
        )
        tail_log_probability = jax.scipy.special.logsumexp(
            jnp.where(complement, log_probs, -jnp.inf)
        )
        teacher_active = jnp.asarray(target.probabilities, dtype=logits.dtype)
        teacher_tail = jnp.asarray(target.aggregate_tail_mass, dtype=logits.dtype)
        active_terms = jnp.where(
            teacher_active > 0, -teacher_active * active_log_probs, 0.0
        )
        tail_term = jnp.where(
            teacher_tail > 0, -teacher_tail * tail_log_probability, 0.0
        )
        losses.append(jnp.sum(active_terms) + tail_term)
    loss = jnp.mean(jnp.stack(losses))
    return loss, {"exemplar.coarse_cross_entropy": loss}


def _validate_logits(logits: Any, batch_size: int, sequence_length: int) -> None:
    shape = getattr(logits, "shape", None)
    if (
        len(shape or ()) != 3
        or shape[0] != batch_size
        or shape[1] != sequence_length
        or shape[2] < 2
    ):
        raise BehavioralObjectiveError(
            "logits must be [batch, sequence, vocabulary>=2]"
        )


def _validate_corridor_inputs(
    logits: Any, batch: CorridorBatchV1, policy: BehavioralObjectivePolicyV1
) -> None:
    if not isinstance(batch, CorridorBatchV1) or not isinstance(
        policy, BehavioralObjectivePolicyV1
    ):
        raise BehavioralObjectiveError(
            "corridor objective requires neutral batch and policy"
        )
    _validate_logits(logits, batch.input_ids.shape[0], batch.input_ids.shape[1])
    if not (
        len(batch.example_indices)
        == len(batch.positions)
        == len(batch.mode_ids)
        == len(batch.assignment_weights)
    ):
        raise BehavioralObjectiveError("corridor assignment tensors disagree")
    rows = np.unique(batch.example_indices)
    if (
        len(rows) > batch.input_ids.shape[0]
        or np.any(batch.positions < 0)
        or np.any(batch.positions >= batch.input_ids.shape[1])
    ):
        raise BehavioralObjectiveError("corridor coordinates are invalid")
    if not set(batch.mode_ids.tolist()).issubset(
        set(batch.mode_bounds.statistics_by_mode)
    ):
        raise BehavioralObjectiveError("corridor mode has no declared statistics")
    local_rows = np.searchsorted(rows, batch.example_indices)
    effective = (
        batch.assignment_weights * batch.attention_mask[local_rows, batch.positions]
    )
    if (
        not np.all(np.isfinite(effective))
        or np.any(effective < 0)
        or float(np.sum(effective)) <= 0
    ):
        raise BehavioralObjectiveError(
            "corridor effective assignment weight must be positive"
        )


def _corridor_bounds(batch: CorridorBatchV1) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for mode in batch.mode_ids.tolist():
        declaration = batch.mode_bounds.statistics_by_mode.get(int(mode))
        if declaration is None or set(declaration) != set(_STATISTICS):
            raise BehavioralObjectiveError("corridor mode statistics are incomplete")
        lower.append([declaration[name].minimum for name in _STATISTICS])
        upper.append([declaration[name].maximum for name in _STATISTICS])
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def _validate_exemplar_inputs(logits: Any, batch: ExemplarBatchV1) -> None:
    if not isinstance(batch, ExemplarBatchV1):
        raise BehavioralObjectiveError("exemplar objective requires a neutral batch")
    _validate_logits(logits, batch.input_ids.shape[0], batch.input_ids.shape[1])
    if len(batch.sparse_targets) != batch.input_ids.shape[0] or len(
        batch.selected_positions
    ) != len(batch.sparse_targets):
        raise BehavioralObjectiveError("exemplar targets and coordinates disagree")
    vocabulary_size = int(logits.shape[-1])
    for position, target in zip(
        batch.selected_positions.tolist(), batch.sparse_targets, strict=True
    ):
        if position < 0 or position >= batch.input_ids.shape[1]:
            raise BehavioralObjectiveError("exemplar position is invalid")
        token_ids, probabilities = target.token_ids, target.probabilities
        if (
            len(token_ids) == 0
            or len(token_ids) >= vocabulary_size
            or len(set(token_ids.tolist())) != len(token_ids)
        ):
            raise BehavioralObjectiveError(
                "exemplar active tokens cannot define an aggregate tail"
            )
        if (
            np.any(token_ids < 0)
            or np.any(token_ids >= vocabulary_size)
            or np.any(probabilities < 0)
        ):
            raise BehavioralObjectiveError("exemplar sparse target is invalid")
        total = float(np.sum(probabilities)) + target.aggregate_tail_mass
        if not np.isfinite(total) or not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
            raise BehavioralObjectiveError("exemplar teacher outcomes must sum to one")


__all__ = [
    "BEHAVIOR_OBJECTIVE_POLICY_V1",
    "BEHAVIOR_OBJECTIVE_REDUCTION_V1",
    "BehavioralObjectiveError",
    "BehavioralObjectivePolicyV1",
    "DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1",
    "corridor_objective_v1",
    "exemplar_coarse_cross_entropy_v1",
]
