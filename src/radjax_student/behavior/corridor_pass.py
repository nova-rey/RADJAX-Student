"""Deterministic, architecture-neutral corridor learning pass (P5.6).

This boundary consumes neutral batches, the admitted P5.3 behavioral authority,
and a caller-owned forward function.  It deliberately contains no archive,
tokenizer, or architecture-plugin knowledge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from radjax_student.artifacts.native_v3_v6 import (
    NativeV3V6BehavioralProjection,
    NativeV3V6ProjectionError,
    _require_admitted_native_v3_v6_projection,
)
from radjax_student.behavior.jax_pass_adapter import (
    execute_behavior_jax_pass_v1,
    jax_tree_digest_payload_v1,
    jax_tree_leaves_v1,
)
from radjax_student.behavior.models import BehavioralBatchesV1, CorridorBatchV1
from radjax_student.behavior.objectives import (
    BEHAVIOR_OBJECTIVE_REDUCTION_V1,
    DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1,
    BehavioralObjectivePolicyV1,
    corridor_objective_v1,
)
from radjax_student.contracts import ParameterTreeLayout
from radjax_student.optimizers import (
    JaxOptimizerState,
    OptimizerConfig,
    SgdOptimizer,
    advanced_jax_optimizer_state,
    apply_verified_jax_updates,
    validate_jax_optimizer_state,
)
from radjax_student.optimizers.errors import OptimizerContractError

CORRIDOR_PASS_ID_V1 = "corridor_v1"
CORRIDOR_ORDERING_POLICY_V1 = "training_example_position_mode.v1"
CORRIDOR_BATCHING_POLICY_V1 = "neutral_full_partition_batch.v1"


class CorridorPassError(ValueError):
    """A corridor pass cannot meet its closed learning contract."""


@dataclass(frozen=True)
class CorridorRunBindingV1:
    """All non-numerical authority required to resume this pass safely."""

    contract_commit: str
    tome_commit: str
    accepted_receipt_identity: str
    language_binding_digest: str
    hf_projection_identity: str
    behavioral_source_identity: str
    behavioral_authority_digest: str
    architecture_config_identity: str
    split_identity: str
    corridor_objective_policy_id: str
    reduction_id: str
    ordering_policy_id: str = CORRIDOR_ORDERING_POLICY_V1
    batching_policy_id: str = CORRIDOR_BATCHING_POLICY_V1

    def __post_init__(self) -> None:
        if (
            self.corridor_objective_policy_id
            != DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1.policy_id
        ):
            raise CorridorPassError("unsupported corridor objective policy")
        if self.reduction_id != BEHAVIOR_OBJECTIVE_REDUCTION_V1:
            raise CorridorPassError("unsupported corridor reduction")
        if self.ordering_policy_id != CORRIDOR_ORDERING_POLICY_V1:
            raise CorridorPassError("unsupported corridor ordering policy")
        if self.batching_policy_id != CORRIDOR_BATCHING_POLICY_V1:
            raise CorridorPassError("unsupported corridor batching policy")
        if any(
            not isinstance(value, str) or not value for value in self.to_dict().values()
        ):
            raise CorridorPassError("corridor binding identities must be nonempty")

    def to_dict(self) -> dict[str, str]:
        return dict(vars(self))

    @property
    def identity(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class CorridorCheckpointV1:
    """In-memory resumable state after exactly one deterministic corridor pass."""

    binding: CorridorRunBindingV1
    parameter_layout: ParameterTreeLayout
    optimizer_state: JaxOptimizerState
    parameters: Any
    epoch: int
    cursor: int
    materialization_identity: str
    canonical_passport_registry_identity: str
    pass_id: str = CORRIDOR_PASS_ID_V1

    def __post_init__(self) -> None:
        if (
            self.pass_id != CORRIDOR_PASS_ID_V1
            or self.epoch < 0
            or self.cursor < 0
            or not self.materialization_identity
            or not self.canonical_passport_registry_identity
        ):
            raise CorridorPassError("corridor checkpoint cursor is invalid")
        if (
            self.optimizer_state.envelope.parameter_paths
            != self.parameter_layout.logical_paths
        ):
            raise CorridorPassError("corridor optimizer state does not match layout")

    @property
    def identity(self) -> str:
        return _digest(
            {
                "binding": self.binding.to_dict(),
                "layout": self.parameter_layout.to_dict(),
                "optimizer_identity": self.optimizer_state.envelope.optimizer_id,
                "optimizer": self.optimizer_state.envelope.to_dict(),
                "optimizer_descriptor": self.optimizer_state.descriptor.to_dict(),
                "optimizer_arrays": _tree_digest(self.optimizer_state.arrays),
                "parameters": _tree_digest(self.parameters),
                "pass_id": self.pass_id,
                "epoch": self.epoch,
                "cursor": self.cursor,
                "materialization_identity": self.materialization_identity,
                "canonical_passport_registry_identity": (
                    self.canonical_passport_registry_identity
                ),
            }
        )


@dataclass(frozen=True)
class CorridorPassResultV1:
    """Finite execution evidence plus the sole P5.6 continuation checkpoint."""

    checkpoint: CorridorCheckpointV1
    loss: float
    gradient_norm: float
    changed_parameter_paths: tuple[str, ...]
    ordered_coordinates: tuple[tuple[str, int, int], ...]


def run_corridor_pass_v1(
    *,
    batch: CorridorBatchV1,
    materialization: BehavioralBatchesV1,
    projection: NativeV3V6BehavioralProjection,
    binding: CorridorRunBindingV1,
    parameters: Any,
    parameter_layout: ParameterTreeLayout,
    optimizer: SgdOptimizer,
    optimizer_config: OptimizerConfig,
    optimizer_state: JaxOptimizerState,
    forward: Callable[[Any, Any, Any], Any],
    epoch: int = 0,
    cursor: int = 0,
    policy: BehavioralObjectivePolicyV1 = DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1,
) -> CorridorPassResultV1:
    """Execute one full training-only corridor batch and emit its checkpoint."""

    if batch.partition != "training":
        raise CorridorPassError("corridor pass refuses held-out batches")
    if binding.split_identity == "":  # keeps the partition contract explicit
        raise CorridorPassError("corridor split identity is required")
    if not isinstance(materialization, BehavioralBatchesV1):
        raise CorridorPassError("corridor pass requires sealed P5.4 materialization")
    registry, registry_identity = _canonical_passport_registry(projection)
    if batch is not materialization.training_corridor:
        raise CorridorPassError("corridor batch is not the sealed training partition")
    if (
        materialization.split.split_identity != binding.split_identity
        or materialization.split.behavioral_source_identity
        != binding.behavioral_source_identity
    ):
        raise CorridorPassError("corridor materialization continuity mismatch")
    projection_matches_source = (
        projection.behavioral_source_identity
        == materialization.split.behavioral_source_identity
        == binding.behavioral_source_identity
    )
    if not projection_matches_source or (
        materialization.descriptor.authoritative_exemplar_passport_keys != registry
        or _materialized_passport_registry(materialization) != registry
    ):
        raise CorridorPassError("corridor materialization passport authority mismatch")
    if optimizer.optimizer_id != optimizer_config.optimizer_id:
        raise CorridorPassError("corridor optimizer configuration mismatch")
    if (
        policy.policy_id != binding.corridor_objective_policy_id
        or policy.reduction_id != binding.reduction_id
    ):
        raise CorridorPassError("corridor objective binding mismatch")
    try:
        validate_jax_optimizer_state(
            optimizer_state,
            optimizer=optimizer,
            optimizer_id=optimizer.optimizer_id,
            parameter_layout=parameter_layout,
            descriptor=optimizer.jax_state_descriptor(parameter_layout),
        )
    except OptimizerContractError as exc:
        raise CorridorPassError(
            "corridor optimizer state does not match supplied optimizer"
        ) from exc

    ordered = _ordered_batch(batch)
    coordinates = tuple(
        (ordered.example_ids[int(row)], int(position), int(mode))
        for row, position, mode in zip(
            ordered.example_indices, ordered.positions, ordered.mode_ids, strict=True
        )
    )

    def objective(logits: Any) -> Any:
        return corridor_objective_v1(logits, ordered, policy=policy)[0]

    computation = execute_behavior_jax_pass_v1(
        parameters=parameters,
        input_ids=ordered.input_ids,
        attention_mask=ordered.attention_mask,
        forward=forward,
        objective=objective,
    )
    loss_value = float(computation.loss)
    gradients = computation.gradients
    gradient_norm = computation.gradient_norm
    if (
        not np.isfinite(loss_value)
        or not np.isfinite(gradient_norm)
        or gradient_norm == 0.0
    ):
        raise CorridorPassError("corridor loss and gradient must be finite and nonzero")
    updated, arrays, changed, _ = apply_verified_jax_updates(
        optimizer=optimizer,
        parameters=parameters,
        gradients=gradients,
        optimizer_array_state=optimizer_state.arrays,
        update_mask=parameter_layout.update_mask(
            parameters, parameter_layout.logical_paths
        ),
        config=optimizer_config,
        schedule_values={"learning_rate": optimizer_config.learning_rate},
    )
    changed_paths = tuple(
        path
        for path, leaf in zip(
            parameter_layout.logical_paths,
            jax_tree_leaves_v1(changed),
            strict=True,
        )
        if bool(leaf)
    )
    if not changed_paths:
        raise CorridorPassError("corridor pass did not change parameters")
    checkpoint = CorridorCheckpointV1(
        binding=binding,
        parameter_layout=parameter_layout,
        optimizer_state=advanced_jax_optimizer_state(optimizer_state, arrays),
        parameters=updated,
        epoch=epoch,
        cursor=cursor + len(coordinates),
        materialization_identity=materialization.descriptor.identity,
        canonical_passport_registry_identity=registry_identity,
    )
    return CorridorPassResultV1(
        checkpoint, loss_value, gradient_norm, changed_paths, coordinates
    )


def replay_corridor_pass_v1(
    *, expected: CorridorCheckpointV1, **kwargs: Any
) -> CorridorPassResultV1:
    """Re-execute the same input and require byte-for-byte checkpoint identity."""

    result = run_corridor_pass_v1(**kwargs)
    if result.checkpoint.identity != expected.identity:
        raise CorridorPassError("corridor checkpoint replay mismatch")
    return result


def _ordered_batch(batch: CorridorBatchV1) -> CorridorBatchV1:
    """Canonicalize assignment traversal without changing the neutral surface."""

    if len(batch.example_ids) != batch.input_ids.shape[0]:
        raise CorridorPassError("corridor examples and tensor rows disagree")
    names = np.asarray([batch.example_ids[int(row)] for row in batch.example_indices])
    order = np.asarray(
        sorted(
            range(len(names)),
            key=lambda index: (
                names[index].encode("utf-8"),
                int(batch.positions[index]),
                int(batch.mode_ids[index]),
            ),
        )
    )
    from dataclasses import replace

    return replace(
        batch,
        example_indices=batch.example_indices[order],
        positions=batch.positions[order],
        mode_ids=batch.mode_ids[order],
        assignment_weights=batch.assignment_weights[order],
    )


def _canonical_passport_registry(
    projection: NativeV3V6BehavioralProjection,
) -> tuple[tuple[tuple[str, int, str], ...], str]:
    """Derive P5.3's complete selected-passport registry at the public entry."""

    if not isinstance(projection, NativeV3V6BehavioralProjection):
        raise CorridorPassError("corridor pass requires admitted P5.3 projection")
    try:
        _require_admitted_native_v3_v6_projection(projection)
    except NativeV3V6ProjectionError as exc:
        raise CorridorPassError(
            "corridor pass requires admitted P5.3 projection"
        ) from exc
    try:
        values = tuple(
            (
                str(passport["selected_example_id"]),
                passport["selected_position"],
                str(passport["corridor_fingerprint_id"]),
            )
            for passport in projection.selected_passports
        )
    except (KeyError, TypeError) as exc:
        raise CorridorPassError("P5.3 passport registry is incomplete") from exc
    if not values or any(type(value[1]) is not int for value in values):
        raise CorridorPassError("P5.3 passport registry is invalid")
    canonical = tuple(
        sorted(
            values,
            key=lambda value: (
                value[0].encode("utf-8"),
                value[1],
                value[2].encode("utf-8"),
            ),
        )
    )
    if len(set(canonical)) != len(canonical):
        raise CorridorPassError("P5.3 passport registry is not unique")
    return canonical, _digest({"passports": canonical})


def _materialized_passport_registry(
    materialization: BehavioralBatchesV1,
) -> tuple[tuple[str, int, str], ...]:
    try:
        values = tuple(
            (
                str(passport["selected_example_id"]),
                passport["selected_position"],
                str(passport["corridor_fingerprint_id"]),
            )
            for batch in (
                materialization.training_exemplars,
                materialization.held_out_exemplars,
            )
            for passport in batch.passports
        )
    except (KeyError, TypeError) as exc:
        raise CorridorPassError("materialized passport registry is incomplete") from exc
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value[0].encode("utf-8"),
                value[1],
                value[2].encode("utf-8"),
            ),
        )
    )


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _tree_digest(value: Any) -> str:
    return _digest(jax_tree_digest_payload_v1(value))
