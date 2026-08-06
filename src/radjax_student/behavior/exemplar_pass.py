"""Deterministic exemplar continuation pass and final Phase 5 checkpoint.

This boundary consumes the neutral P5.4 exemplar batch only after an accepted
P5.6 corridor checkpoint.  It intentionally has no artifact, archive,
locator, tokenizer, or architecture-plugin knowledge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from radjax_student.behavior.corridor_pass import CorridorCheckpointV1
from radjax_student.behavior.jax_pass_adapter import (
    execute_behavior_jax_pass_v1,
    jax_tree_digest_payload_v1,
    jax_tree_leaves_v1,
)
from radjax_student.behavior.models import ExemplarBatchV1
from radjax_student.behavior.objectives import exemplar_coarse_cross_entropy_v1
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

EXEMPLAR_PASS_ID_V1 = "exemplar_v1"
EXEMPLAR_OBJECTIVE_POLICY_V1 = "exemplar_coarse_cross_entropy_v1"
EXEMPLAR_ORDERING_POLICY_V1 = "passport_example_position_fingerprint.v1"
EXEMPLAR_BATCHING_POLICY_V1 = "neutral_full_partition_batch.v1"


class ExemplarPassError(ValueError):
    """An exemplar continuation cannot meet its closed learning contract."""


@dataclass(frozen=True)
class ExemplarRunBindingV1:
    """All authority that must continue unchanged from the corridor pass."""

    predecessor_checkpoint_identity: str
    contract_commit: str
    tome_commit: str
    accepted_receipt_identity: str
    language_binding_digest: str
    hf_projection_identity: str
    behavioral_source_identity: str
    behavioral_authority_digest: str
    architecture_config_identity: str
    split_identity: str
    materialization_identity: str
    canonical_passport_registry_identity: str
    corridor_objective_policy_id: str
    reduction_id: str
    corridor_ordering_policy_id: str
    corridor_batching_policy_id: str
    exemplar_objective_policy_id: str = EXEMPLAR_OBJECTIVE_POLICY_V1
    ordering_policy_id: str = EXEMPLAR_ORDERING_POLICY_V1
    batching_policy_id: str = EXEMPLAR_BATCHING_POLICY_V1

    def __post_init__(self) -> None:
        if self.exemplar_objective_policy_id != EXEMPLAR_OBJECTIVE_POLICY_V1:
            raise ExemplarPassError("unsupported exemplar objective policy")
        if self.ordering_policy_id != EXEMPLAR_ORDERING_POLICY_V1:
            raise ExemplarPassError("unsupported exemplar ordering policy")
        if self.batching_policy_id != EXEMPLAR_BATCHING_POLICY_V1:
            raise ExemplarPassError("unsupported exemplar batching policy")
        if any(
            not isinstance(value, str) or not value for value in self.to_dict().values()
        ):
            raise ExemplarPassError("exemplar binding identities must be nonempty")

    @classmethod
    def from_corridor_checkpoint(
        cls, checkpoint: CorridorCheckpointV1
    ) -> ExemplarRunBindingV1:
        """Create the only permissible continuation authority from P5.6."""

        binding = checkpoint.binding
        return cls(
            predecessor_checkpoint_identity=checkpoint.identity,
            contract_commit=binding.contract_commit,
            tome_commit=binding.tome_commit,
            accepted_receipt_identity=binding.accepted_receipt_identity,
            language_binding_digest=binding.language_binding_digest,
            hf_projection_identity=binding.hf_projection_identity,
            behavioral_source_identity=binding.behavioral_source_identity,
            behavioral_authority_digest=binding.behavioral_authority_digest,
            architecture_config_identity=binding.architecture_config_identity,
            split_identity=binding.split_identity,
            materialization_identity=checkpoint.materialization_identity,
            canonical_passport_registry_identity=(
                checkpoint.canonical_passport_registry_identity
            ),
            corridor_objective_policy_id=binding.corridor_objective_policy_id,
            reduction_id=binding.reduction_id,
            corridor_ordering_policy_id=binding.ordering_policy_id,
            corridor_batching_policy_id=binding.batching_policy_id,
        )

    def to_dict(self) -> dict[str, str]:
        return dict(vars(self))

    @property
    def identity(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class ExemplarCheckpointV1:
    """Final sequential state after a corridor then exemplar pass."""

    binding: ExemplarRunBindingV1
    parameter_layout: ParameterTreeLayout
    optimizer_state: JaxOptimizerState
    parameters: Any
    epoch: int
    cursor: int
    pass_id: str = EXEMPLAR_PASS_ID_V1

    def __post_init__(self) -> None:
        if self.pass_id != EXEMPLAR_PASS_ID_V1 or self.epoch < 0 or self.cursor < 0:
            raise ExemplarPassError("exemplar checkpoint cursor is invalid")
        if (
            self.optimizer_state.envelope.parameter_paths
            != self.parameter_layout.logical_paths
        ):
            raise ExemplarPassError("exemplar optimizer state does not match layout")

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
            }
        )


@dataclass(frozen=True)
class ExemplarPassResultV1:
    """Finite evidence and the sole final continuation checkpoint."""

    checkpoint: ExemplarCheckpointV1
    loss: float
    gradient_norm: float
    changed_parameter_paths: tuple[str, ...]
    ordered_passport_keys: tuple[tuple[str, int, str], ...]


def run_exemplar_pass_v1(
    *,
    predecessor: CorridorCheckpointV1,
    batch: ExemplarBatchV1,
    binding: ExemplarRunBindingV1,
    parameter_layout: ParameterTreeLayout,
    optimizer: SgdOptimizer,
    optimizer_config: OptimizerConfig,
    forward: Callable[[Any, Any, Any], Any],
    epoch: int = 0,
    cursor: int = 0,
) -> ExemplarPassResultV1:
    """Resume P5.6 once, execute all training exemplars, and checkpoint."""

    if not isinstance(batch, ExemplarBatchV1):
        raise ExemplarPassError("exemplar pass requires a neutral exemplar batch")
    if batch.partition != "training":
        raise ExemplarPassError("exemplar pass refuses held-out batches")
    _validate_predecessor(predecessor, binding, parameter_layout)
    if optimizer.optimizer_id != optimizer_config.optimizer_id:
        raise ExemplarPassError("exemplar optimizer configuration mismatch")
    try:
        validate_jax_optimizer_state(
            predecessor.optimizer_state,
            optimizer=optimizer,
            optimizer_id=optimizer.optimizer_id,
            parameter_layout=parameter_layout,
            descriptor=optimizer.jax_state_descriptor(parameter_layout),
        )
    except OptimizerContractError as exc:
        raise ExemplarPassError(
            "exemplar optimizer state does not match supplied optimizer"
        ) from exc

    ordered_keys = _validate_canonical_passports(batch)
    computation = execute_behavior_jax_pass_v1(
        parameters=predecessor.parameters,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        forward=forward,
        objective=lambda logits: exemplar_coarse_cross_entropy_v1(logits, batch)[0],
    )
    loss_value = float(computation.loss)
    gradients = computation.gradients
    gradient_norm = computation.gradient_norm
    if (
        not np.isfinite(loss_value)
        or not np.isfinite(gradient_norm)
        or gradient_norm == 0.0
    ):
        raise ExemplarPassError("exemplar loss and gradient must be finite and nonzero")
    updated, arrays, changed, _ = apply_verified_jax_updates(
        optimizer=optimizer,
        parameters=predecessor.parameters,
        gradients=gradients,
        optimizer_array_state=predecessor.optimizer_state.arrays,
        update_mask=parameter_layout.update_mask(
            predecessor.parameters, parameter_layout.logical_paths
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
        raise ExemplarPassError("exemplar pass did not change parameters")
    checkpoint = ExemplarCheckpointV1(
        binding=binding,
        parameter_layout=parameter_layout,
        optimizer_state=advanced_jax_optimizer_state(
            predecessor.optimizer_state, arrays
        ),
        parameters=updated,
        epoch=epoch,
        cursor=cursor + len(ordered_keys),
    )
    return ExemplarPassResultV1(
        checkpoint, loss_value, gradient_norm, changed_paths, ordered_keys
    )


def replay_exemplar_pass_v1(
    *, expected: ExemplarCheckpointV1, **kwargs: Any
) -> ExemplarPassResultV1:
    """Re-execute the exact continuation and require the same final identity."""

    result = run_exemplar_pass_v1(**kwargs)
    if result.checkpoint.identity != expected.identity:
        raise ExemplarPassError("exemplar checkpoint replay mismatch")
    return result


def _validate_predecessor(
    predecessor: CorridorCheckpointV1,
    binding: ExemplarRunBindingV1,
    layout: ParameterTreeLayout,
) -> None:
    if not isinstance(predecessor, CorridorCheckpointV1):
        raise ExemplarPassError("exemplar pass requires a P5.6 corridor predecessor")
    if predecessor.identity != binding.predecessor_checkpoint_identity:
        raise ExemplarPassError("exemplar predecessor checkpoint identity mismatch")
    if predecessor.parameter_layout != layout:
        raise ExemplarPassError("exemplar predecessor layout mismatch")
    continued = {
        "contract_commit": predecessor.binding.contract_commit,
        "tome_commit": predecessor.binding.tome_commit,
        "accepted_receipt_identity": predecessor.binding.accepted_receipt_identity,
        "language_binding_digest": predecessor.binding.language_binding_digest,
        "hf_projection_identity": predecessor.binding.hf_projection_identity,
        "behavioral_source_identity": predecessor.binding.behavioral_source_identity,
        "behavioral_authority_digest": predecessor.binding.behavioral_authority_digest,
        "architecture_config_identity": (
            predecessor.binding.architecture_config_identity
        ),
        "split_identity": predecessor.binding.split_identity,
        "materialization_identity": predecessor.materialization_identity,
        "canonical_passport_registry_identity": (
            predecessor.canonical_passport_registry_identity
        ),
        "corridor_objective_policy_id": (
            predecessor.binding.corridor_objective_policy_id
        ),
        "reduction_id": predecessor.binding.reduction_id,
        "corridor_ordering_policy_id": predecessor.binding.ordering_policy_id,
        "corridor_batching_policy_id": predecessor.binding.batching_policy_id,
    }
    if any(getattr(binding, key) != value for key, value in continued.items()):
        raise ExemplarPassError("exemplar continuation authority mismatch")


def _validate_canonical_passports(
    batch: ExemplarBatchV1,
) -> tuple[tuple[str, int, str], ...]:
    if len(batch.passports) != len(batch.sparse_targets):
        raise ExemplarPassError("exemplar passports and targets disagree")
    keys: list[tuple[str, int, str]] = []
    for passport in batch.passports:
        try:
            key = (
                str(passport["selected_example_id"]),
                passport["selected_position"],
                str(passport["corridor_fingerprint_id"]),
            )
        except (KeyError, TypeError) as exc:
            raise ExemplarPassError("exemplar passport is incomplete") from exc
        if type(key[1]) is not int:
            raise ExemplarPassError("exemplar passport coordinate is invalid")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ExemplarPassError("exemplar passports are not unique")
    canonical = tuple(
        sorted(
            keys,
            key=lambda item: (
                item[0].encode("utf-8"),
                item[1],
                item[2].encode("utf-8"),
            ),
        )
    )
    if tuple(keys) != canonical:
        raise ExemplarPassError("exemplar passports are reordered")
    return canonical


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _tree_digest(value: Any) -> str:
    return _digest(jax_tree_digest_payload_v1(value))
