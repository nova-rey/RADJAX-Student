"""Immutable, architecture-neutral behavioral materialization values."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class BehaviorSplitV1:
    """The complete, identity-bound result of ``BehaviorSplitPolicyV1``."""

    policy_id: str
    rule_version: str
    behavioral_source_identity: str
    assignments: Mapping[str, str]
    split_identity: str

    @property
    def training_example_ids(self) -> tuple[str, ...]:
        return tuple(
            key for key, value in self.assignments.items() if value == "training"
        )

    @property
    def held_out_example_ids(self) -> tuple[str, ...]:
        return tuple(
            key for key, value in self.assignments.items() if value == "held_out"
        )


@dataclass(frozen=True)
class ModeStatisticBoundsV1:
    """One Contract-declared inclusive statistic interval for a mode."""

    minimum: float
    mean: float
    maximum: float


@dataclass(frozen=True)
class ModeBoundsV1:
    """Complete Contract-declared mode/statistic bounds without model semantics."""

    minimum_mode_id: int
    maximum_mode_id: int
    declared_mode_ids: tuple[int, ...]
    declarations: tuple[Mapping[str, object], ...]
    statistics_by_mode: Mapping[int, Mapping[str, ModeStatisticBoundsV1]]


@dataclass(frozen=True)
class CorridorBatchV1:
    """Neutral corridor tensors for exactly one leakage-free partition."""

    partition: str
    example_ids: tuple[str, ...]
    input_ids: np.ndarray
    attention_mask: np.ndarray
    example_indices: np.ndarray
    positions: np.ndarray
    mode_ids: np.ndarray
    mode_bounds: ModeBoundsV1
    assignment_weights: np.ndarray


@dataclass(frozen=True)
class SparseTargetV1:
    """One declared active sparse target distribution and its tail mass."""

    token_ids: np.ndarray
    probabilities: np.ndarray
    aggregate_tail_mass: float


@dataclass(frozen=True)
class ExemplarBatchV1:
    """Neutral selected-exemplar values for exactly one partition."""

    partition: str
    example_ids: tuple[str, ...]
    input_ids: np.ndarray
    attention_mask: np.ndarray
    selected_example_indices: np.ndarray
    selected_positions: np.ndarray
    sparse_targets: tuple[SparseTargetV1, ...]
    passports: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class BehavioralMaterializationDescriptorV1:
    """Canonical complete P5.4 partition authority, independent of delivery."""

    policy_id: str
    rule_version: str
    behavioral_source_identity: str
    split_identity: str
    assignment_identity: str
    training_corridor_identity: str
    held_out_corridor_identity: str
    training_exemplar_identity: str
    held_out_exemplar_identity: str
    authoritative_exemplar_passport_keys: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in vars(self).values()
            if not isinstance(value, tuple)
        ):
            raise ValueError("behavioral materialization descriptor is incomplete")

    def to_dict(self) -> dict[str, str]:
        return dict(vars(self))

    @property
    def identity(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class BehavioralBatchesV1:
    """P5.4 output: split plus neutral batches, and nothing delivery-specific."""

    split: BehaviorSplitV1
    training_corridor: CorridorBatchV1
    held_out_corridor: CorridorBatchV1
    training_exemplars: ExemplarBatchV1
    held_out_exemplars: ExemplarBatchV1
    descriptor: BehavioralMaterializationDescriptorV1

    def __post_init__(self) -> None:
        _validate_partition_batches(self)
        if self.descriptor != _behavioral_materialization_descriptor_v1(
            split=self.split,
            training_corridor=self.training_corridor,
            held_out_corridor=self.held_out_corridor,
            training_exemplars=self.training_exemplars,
            held_out_exemplars=self.held_out_exemplars,
            authoritative_exemplar_passport_keys=(
                self.descriptor.authoritative_exemplar_passport_keys
            ),
        ):
            raise ValueError("behavioral materialization descriptor mismatch")


def _behavioral_materialization_descriptor_v1(
    *,
    split: BehaviorSplitV1,
    training_corridor: CorridorBatchV1,
    held_out_corridor: CorridorBatchV1,
    training_exemplars: ExemplarBatchV1,
    held_out_exemplars: ExemplarBatchV1,
    authoritative_exemplar_passport_keys: tuple[tuple[str, int, str], ...],
) -> BehavioralMaterializationDescriptorV1:
    """Derive the immutable complete P5.4 authority descriptor."""

    return BehavioralMaterializationDescriptorV1(
        policy_id=split.policy_id,
        rule_version=split.rule_version,
        behavioral_source_identity=split.behavioral_source_identity,
        split_identity=split.split_identity,
        assignment_identity=_digest({"assignments": sorted(split.assignments.items())}),
        training_corridor_identity=_digest(
            {"coordinates": _coordinates(training_corridor)}
        ),
        held_out_corridor_identity=_digest(
            {"coordinates": _coordinates(held_out_corridor)}
        ),
        training_exemplar_identity=_digest(
            {"passports": _passport_keys(training_exemplars)}
        ),
        held_out_exemplar_identity=_digest(
            {"passports": _passport_keys(held_out_exemplars)}
        ),
        authoritative_exemplar_passport_keys=authoritative_exemplar_passport_keys,
    )


def _validate_partition_batches(batches: BehavioralBatchesV1) -> None:
    corridor = (batches.training_corridor, batches.held_out_corridor)
    exemplars = (batches.training_exemplars, batches.held_out_exemplars)
    if tuple(item.partition for item in corridor + exemplars) != (
        "training",
        "held_out",
        "training",
        "held_out",
    ):
        raise ValueError("behavioral batch partitions are invalid")
    if set(batches.training_corridor.example_ids) != set(
        batches.split.training_example_ids
    ):
        raise ValueError("training corridor does not cover the split")
    if set(batches.held_out_corridor.example_ids) != set(
        batches.split.held_out_example_ids
    ):
        raise ValueError("held-out corridor does not cover the split")
    if not set(batches.training_exemplars.example_ids) <= set(
        batches.split.training_example_ids
    ) or not set(batches.held_out_exemplars.example_ids) <= set(
        batches.split.held_out_example_ids
    ):
        raise ValueError("exemplar batch violates the split")
    training_keys = _passport_keys(batches.training_exemplars)
    held_out_keys = _passport_keys(batches.held_out_exemplars)
    all_keys = training_keys + held_out_keys
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("exemplar passports are not unique")
    expected_training = tuple(
        key
        for key in batches.descriptor.authoritative_exemplar_passport_keys
        if batches.split.assignments[key[0]] == "training"
    )
    expected_held_out = tuple(
        key
        for key in batches.descriptor.authoritative_exemplar_passport_keys
        if batches.split.assignments[key[0]] == "held_out"
    )
    if training_keys != expected_training or held_out_keys != expected_held_out:
        raise ValueError("exemplar batches do not cover policy assignments")


def _coordinates(batch: CorridorBatchV1) -> tuple[tuple[str, int, int], ...]:
    global_rows = tuple(sorted(set(int(row) for row in batch.example_indices)))
    if len(batch.example_ids) != batch.input_ids.shape[0] or len(global_rows) != len(
        batch.example_ids
    ):
        raise ValueError("corridor batch is incomplete")
    ids = dict(zip(global_rows, batch.example_ids, strict=True))
    return tuple(
        sorted(
            (
                (ids[int(row)], int(position), int(mode))
                for row, position, mode in zip(
                    batch.example_indices, batch.positions, batch.mode_ids, strict=True
                )
            ),
            key=lambda item: (item[0].encode("utf-8"), item[1], item[2]),
        )
    )


def _passport_keys(batch: ExemplarBatchV1) -> tuple[tuple[str, int, str], ...]:
    if len(batch.passports) != len(batch.sparse_targets):
        raise ValueError("exemplar batch is incomplete")
    try:
        values = tuple(
            (
                str(passport["selected_example_id"]),
                passport["selected_position"],
                str(passport["corridor_fingerprint_id"]),
            )
            for passport in batch.passports
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("exemplar passport is incomplete") from exc
    if any(type(value[1]) is not int for value in values):
        raise ValueError("exemplar passport coordinate is invalid")
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item[0].encode("utf-8"),
                item[1],
                item[2].encode("utf-8"),
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


def frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Recursively preserve record data without retaining mutable producer state."""

    return MappingProxyType(
        {str(key): freeze_value(item) for key, item in value.items()}
    )


def freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return frozen_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    return value
