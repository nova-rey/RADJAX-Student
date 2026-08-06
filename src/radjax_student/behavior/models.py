"""Immutable, architecture-neutral behavioral materialization values."""

from __future__ import annotations

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
class BehavioralBatchesV1:
    """P5.4 output: split plus neutral batches, and nothing delivery-specific."""

    split: BehaviorSplitV1
    training_corridor: CorridorBatchV1
    held_out_corridor: CorridorBatchV1
    training_exemplars: ExemplarBatchV1
    held_out_exemplars: ExemplarBatchV1


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
