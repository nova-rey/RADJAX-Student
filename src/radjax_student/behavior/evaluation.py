"""Deterministic, held-out-only behavioral evaluation (P5.8).

This boundary receives neutral batches, immutable checkpoints, and a
caller-owned forward function.  It deliberately has no optimizer, artifact,
archive, locator, tokenizer, or architecture-plugin knowledge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from radjax_student.behavior.corridor_pass import CorridorCheckpointV1
from radjax_student.behavior.exemplar_pass import (
    EXEMPLAR_PASS_ID_V1,
    ExemplarCheckpointV1,
)
from radjax_student.behavior.jax_pass_adapter import materialize_behavior_jax_inputs_v1
from radjax_student.behavior.models import (
    BehavioralBatchesV1,
    CorridorBatchV1,
    ExemplarBatchV1,
)
from radjax_student.behavior.objectives import (
    DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1,
    corridor_objective_v1,
    exemplar_coarse_cross_entropy_v1,
)

HELD_OUT_EVALUATION_ID_V1 = "held_out_evaluation_v1"


class HeldOutEvaluationError(ValueError):
    """Held-out evidence cannot meet the closed P5.8 contract."""


@dataclass(frozen=True)
class HeldOutEvaluationBindingV1:
    """The complete identity continuity required to evaluate P5.7 state."""

    final_checkpoint_identity: str
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
    corridor_objective_policy_id: str
    reduction_id: str
    corridor_ordering_policy_id: str
    corridor_batching_policy_id: str
    exemplar_objective_policy_id: str
    exemplar_ordering_policy_id: str
    exemplar_batching_policy_id: str
    materialization_identity: str
    canonical_passport_registry_identity: str
    expected_held_out_corridor_identity: str
    expected_held_out_exemplar_identity: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value for value in self.to_dict().values()
        ):
            raise HeldOutEvaluationError(
                "evaluation binding identities must be nonempty"
            )

    @classmethod
    def from_final_checkpoint(
        cls, checkpoint: ExemplarCheckpointV1, expected_batches: BehavioralBatchesV1
    ) -> HeldOutEvaluationBindingV1:
        """Create the only permissible P5.8 continuity binding."""

        binding = checkpoint.binding
        if expected_batches.descriptor.identity != binding.materialization_identity:
            raise HeldOutEvaluationError("expected materialization identity mismatch")
        return cls(
            final_checkpoint_identity=checkpoint.identity,
            predecessor_checkpoint_identity=binding.predecessor_checkpoint_identity,
            contract_commit=binding.contract_commit,
            tome_commit=binding.tome_commit,
            accepted_receipt_identity=binding.accepted_receipt_identity,
            language_binding_digest=binding.language_binding_digest,
            hf_projection_identity=binding.hf_projection_identity,
            behavioral_source_identity=binding.behavioral_source_identity,
            behavioral_authority_digest=binding.behavioral_authority_digest,
            architecture_config_identity=binding.architecture_config_identity,
            split_identity=binding.split_identity,
            corridor_objective_policy_id=binding.corridor_objective_policy_id,
            reduction_id=binding.reduction_id,
            corridor_ordering_policy_id=binding.corridor_ordering_policy_id,
            corridor_batching_policy_id=binding.corridor_batching_policy_id,
            exemplar_objective_policy_id=binding.exemplar_objective_policy_id,
            exemplar_ordering_policy_id=binding.ordering_policy_id,
            exemplar_batching_policy_id=binding.batching_policy_id,
            materialization_identity=expected_batches.descriptor.identity,
            canonical_passport_registry_identity=(
                binding.canonical_passport_registry_identity
            ),
            expected_held_out_corridor_identity=_held_out_corridor_identity(
                expected_batches.held_out_corridor
            ),
            expected_held_out_exemplar_identity=_held_out_exemplar_identity(
                expected_batches.held_out_exemplars
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return dict(vars(self))

    @property
    def identity(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class HeldOutEvaluationReportV1:
    """Finite, replayable P5.8 evidence with no learned-state output."""

    binding: HeldOutEvaluationBindingV1
    corridor_coordinates: tuple[tuple[str, int, int], ...]
    exemplar_passport_keys: tuple[tuple[str, int, str], ...]
    corridor_metrics: tuple[tuple[str, float], ...]
    exemplar_metrics: tuple[tuple[str, float], ...]
    evaluation_id: str = HELD_OUT_EVALUATION_ID_V1

    def __post_init__(self) -> None:
        if self.evaluation_id != HELD_OUT_EVALUATION_ID_V1:
            raise HeldOutEvaluationError("unsupported held-out evaluation identity")

    @property
    def identity(self) -> str:
        return _digest(
            {
                "binding": self.binding.to_dict(),
                "corridor_coordinates": self.corridor_coordinates,
                "exemplar_passport_keys": self.exemplar_passport_keys,
                "corridor_metrics": self.corridor_metrics,
                "exemplar_metrics": self.exemplar_metrics,
                "evaluation_id": self.evaluation_id,
            }
        )


def evaluate_held_out_behavior_v1(
    *,
    corridor_checkpoint: CorridorCheckpointV1,
    final_checkpoint: ExemplarCheckpointV1,
    binding: HeldOutEvaluationBindingV1,
    expected_batches: BehavioralBatchesV1,
    training_corridor: CorridorBatchV1,
    training_exemplars: ExemplarBatchV1,
    held_out_corridor: CorridorBatchV1,
    held_out_exemplars: ExemplarBatchV1,
    forward: Callable[[Any, Any, Any], Any],
) -> HeldOutEvaluationReportV1:
    """Evaluate every held-out coordinate and exemplar exactly once, without updates."""

    _validate_continuity(corridor_checkpoint, final_checkpoint, binding)
    _validate_expected_batches(expected_batches, binding)
    _validate_partitions(
        training_corridor,
        training_exemplars,
        held_out_corridor,
        held_out_exemplars,
        corridor_checkpoint,
        final_checkpoint,
    )
    coordinates = _held_out_coordinates(held_out_corridor)
    passports = _held_out_passports(held_out_exemplars)
    if (
        _digest({"coordinates": coordinates})
        != binding.expected_held_out_corridor_identity
    ):
        raise HeldOutEvaluationError(
            "held-out corridor evidence is incomplete or substituted"
        )
    if _digest({"passports": passports}) != binding.expected_held_out_exemplar_identity:
        raise HeldOutEvaluationError(
            "held-out exemplar evidence is incomplete or substituted"
        )
    final_identity = final_checkpoint.identity
    corridor_inputs = materialize_behavior_jax_inputs_v1(
        held_out_corridor.input_ids, held_out_corridor.attention_mask
    )
    corridor_logits = forward(final_checkpoint.parameters, *corridor_inputs)
    _, corridor_metrics = corridor_objective_v1(
        corridor_logits,
        held_out_corridor,
        policy=DEFAULT_BEHAVIORAL_OBJECTIVE_POLICY_V1,
    )
    exemplar_inputs = materialize_behavior_jax_inputs_v1(
        held_out_exemplars.input_ids, held_out_exemplars.attention_mask
    )
    exemplar_logits = forward(final_checkpoint.parameters, *exemplar_inputs)
    _, exemplar_metrics = exemplar_coarse_cross_entropy_v1(
        exemplar_logits, held_out_exemplars
    )
    if final_checkpoint.identity != final_identity:
        raise HeldOutEvaluationError("evaluation mutated final checkpoint")
    return HeldOutEvaluationReportV1(
        binding=binding,
        corridor_coordinates=coordinates,
        exemplar_passport_keys=passports,
        corridor_metrics=_finite_metrics(corridor_metrics),
        exemplar_metrics=_finite_metrics(exemplar_metrics),
    )


def replay_held_out_evaluation_v1(
    *, expected: HeldOutEvaluationReportV1, **kwargs: Any
) -> HeldOutEvaluationReportV1:
    """Re-evaluate the exact final state and require the identical report."""

    report = evaluate_held_out_behavior_v1(**kwargs)
    if report.identity != expected.identity:
        raise HeldOutEvaluationError("held-out evaluation replay mismatch")
    return report


def _validate_continuity(
    corridor: CorridorCheckpointV1,
    final: ExemplarCheckpointV1,
    binding: HeldOutEvaluationBindingV1,
) -> None:
    if not isinstance(corridor, CorridorCheckpointV1):
        raise HeldOutEvaluationError("evaluation requires a P5.6 corridor checkpoint")
    if (
        not isinstance(final, ExemplarCheckpointV1)
        or final.pass_id != EXEMPLAR_PASS_ID_V1
    ):
        raise HeldOutEvaluationError("evaluation requires a final P5.7 checkpoint")
    if final.identity != binding.final_checkpoint_identity:
        raise HeldOutEvaluationError("final checkpoint identity mismatch")
    if corridor.identity != binding.predecessor_checkpoint_identity:
        raise HeldOutEvaluationError("predecessor checkpoint identity mismatch")
    if final.binding.predecessor_checkpoint_identity != corridor.identity:
        raise HeldOutEvaluationError("final checkpoint predecessor mismatch")
    if (
        final.binding.canonical_passport_registry_identity
        != corridor.canonical_passport_registry_identity
    ):
        raise HeldOutEvaluationError("passport registry lineage mismatch")
    continued = binding.to_dict().copy()
    del continued["final_checkpoint_identity"]
    del continued["expected_held_out_corridor_identity"]
    del continued["expected_held_out_exemplar_identity"]
    expected = {
        "predecessor_checkpoint_identity": (
            final.binding.predecessor_checkpoint_identity
        ),
        "contract_commit": final.binding.contract_commit,
        "tome_commit": final.binding.tome_commit,
        "accepted_receipt_identity": final.binding.accepted_receipt_identity,
        "language_binding_digest": final.binding.language_binding_digest,
        "hf_projection_identity": final.binding.hf_projection_identity,
        "behavioral_source_identity": final.binding.behavioral_source_identity,
        "behavioral_authority_digest": final.binding.behavioral_authority_digest,
        "architecture_config_identity": final.binding.architecture_config_identity,
        "split_identity": final.binding.split_identity,
        "materialization_identity": final.binding.materialization_identity,
        "canonical_passport_registry_identity": (
            final.binding.canonical_passport_registry_identity
        ),
        "corridor_objective_policy_id": final.binding.corridor_objective_policy_id,
        "reduction_id": final.binding.reduction_id,
        "corridor_ordering_policy_id": final.binding.corridor_ordering_policy_id,
        "corridor_batching_policy_id": final.binding.corridor_batching_policy_id,
        "exemplar_objective_policy_id": final.binding.exemplar_objective_policy_id,
        "exemplar_ordering_policy_id": final.binding.ordering_policy_id,
        "exemplar_batching_policy_id": final.binding.batching_policy_id,
    }
    if continued != expected:
        raise HeldOutEvaluationError("evaluation continuity authority mismatch")


def _validate_partitions(
    training_corridor: CorridorBatchV1,
    training_exemplars: ExemplarBatchV1,
    held_out_corridor: CorridorBatchV1,
    held_out_exemplars: ExemplarBatchV1,
    corridor_checkpoint: CorridorCheckpointV1,
    final_checkpoint: ExemplarCheckpointV1,
) -> None:
    if not all(
        isinstance(item, CorridorBatchV1)
        for item in (training_corridor, held_out_corridor)
    ):
        raise HeldOutEvaluationError("evaluation requires neutral corridor batches")
    if not all(
        isinstance(item, ExemplarBatchV1)
        for item in (training_exemplars, held_out_exemplars)
    ):
        raise HeldOutEvaluationError("evaluation requires neutral exemplar batches")
    if (
        training_corridor.partition != "training"
        or training_exemplars.partition != "training"
    ):
        raise HeldOutEvaluationError(
            "evaluation training batches must be training partition"
        )
    if (
        held_out_corridor.partition != "held_out"
        or held_out_exemplars.partition != "held_out"
    ):
        raise HeldOutEvaluationError("evaluation requires held-out batches")
    held_out_ids = set(held_out_corridor.example_ids) | set(
        held_out_exemplars.example_ids
    )
    training_ids = set(training_corridor.example_ids) | set(
        training_exemplars.example_ids
    )
    if not held_out_ids or held_out_ids & training_ids:
        raise HeldOutEvaluationError("held-out evidence leaks into training batches")
    if corridor_checkpoint.cursor != len(_coordinates(training_corridor)):
        raise HeldOutEvaluationError("corridor training cursor is incomplete")
    if final_checkpoint.cursor != len(_passport_keys(training_exemplars)):
        raise HeldOutEvaluationError("exemplar training cursor is incomplete")


def _validate_expected_batches(
    expected_batches: BehavioralBatchesV1, binding: HeldOutEvaluationBindingV1
) -> None:
    if not isinstance(expected_batches, BehavioralBatchesV1):
        raise HeldOutEvaluationError("evaluation requires complete expected batches")
    if expected_batches.descriptor.identity != binding.materialization_identity:
        raise HeldOutEvaluationError("expected materialization identity mismatch")
    if (
        expected_batches.split.split_identity != binding.split_identity
        or expected_batches.split.behavioral_source_identity
        != binding.behavioral_source_identity
    ):
        raise HeldOutEvaluationError("expected split authority mismatch")
    if (
        _held_out_corridor_identity(expected_batches.held_out_corridor)
        != binding.expected_held_out_corridor_identity
        or _held_out_exemplar_identity(expected_batches.held_out_exemplars)
        != binding.expected_held_out_exemplar_identity
    ):
        raise HeldOutEvaluationError("expected held-out evidence identity mismatch")


def _held_out_corridor_identity(batch: CorridorBatchV1) -> str:
    return _digest({"coordinates": _held_out_coordinates(batch)})


def _held_out_exemplar_identity(batch: ExemplarBatchV1) -> str:
    return _digest({"passports": _held_out_passports(batch)})


def _held_out_coordinates(batch: CorridorBatchV1) -> tuple[tuple[str, int, int], ...]:
    coordinates = _coordinates(batch)
    if len(set(coordinates)) != len(coordinates):
        raise HeldOutEvaluationError("held-out corridor coordinates are not unique")
    return coordinates


def _coordinates(batch: CorridorBatchV1) -> tuple[tuple[str, int, int], ...]:
    if len(batch.example_ids) != batch.input_ids.shape[0]:
        raise HeldOutEvaluationError("corridor examples and tensor rows disagree")
    global_rows = tuple(sorted(set(int(row) for row in batch.example_indices)))
    if len(global_rows) != len(batch.example_ids):
        raise HeldOutEvaluationError("corridor examples and coordinates disagree")
    example_id_by_global_row = dict(zip(global_rows, batch.example_ids, strict=True))
    try:
        values = tuple(
            (example_id_by_global_row[int(row)], int(position), int(mode))
            for row, position, mode in zip(
                batch.example_indices, batch.positions, batch.mode_ids, strict=True
            )
        )
    except (IndexError, ValueError) as exc:
        raise HeldOutEvaluationError("corridor coordinate is invalid") from exc
    return tuple(
        sorted(values, key=lambda item: (item[0].encode("utf-8"), item[1], item[2]))
    )


def _held_out_passports(batch: ExemplarBatchV1) -> tuple[tuple[str, int, str], ...]:
    values = _passport_keys(batch)
    if len(set(values)) != len(values):
        raise HeldOutEvaluationError("held-out exemplars are not unique")
    return values


def _passport_keys(batch: ExemplarBatchV1) -> tuple[tuple[str, int, str], ...]:
    if len(batch.passports) != len(batch.sparse_targets):
        raise HeldOutEvaluationError("exemplar passports and targets disagree")
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
        raise HeldOutEvaluationError("exemplar passport is incomplete") from exc
    if any(type(value[1]) is not int for value in values):
        raise HeldOutEvaluationError("exemplar passport coordinate is invalid")
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


def _finite_metrics(metrics: Any) -> tuple[tuple[str, float], ...]:
    values = tuple(sorted((str(key), float(value)) for key, value in metrics.items()))
    if not values or not all(np.isfinite(value) for _, value in values):
        raise HeldOutEvaluationError("held-out metrics must be finite")
    return values


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
