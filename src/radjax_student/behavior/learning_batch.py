"""Finite-JSON, one-source-unit learning batches for neutral behavior values.

This module is deliberately the point at which the sealed P5.4 value model is
reduced to a batch of one.  It knows neither an artifact delivery nor a
checkpoint or architecture.  A source unit carries its behavioral source,
split, and coordinate authority in a content-bound finite-JSON envelope.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from radjax_student.behavior.models import (
    BehavioralBatchesV1,
    CorridorBatchV1,
    ExemplarBatchV1,
)
from radjax_student.contracts._json import json_value
from radjax_student.learning import LearningBatch

BEHAVIOR_SOURCE_UNIT_SCHEMA_V1 = "radjax.student.behavior_source_unit.v1"
CORRIDOR_SOURCE_UNIT_KIND_V1 = "corridor_coordinate"
EXEMPLAR_SOURCE_UNIT_KIND_V1 = "exemplar_passport"

_SOURCE_UNIT_KINDS = frozenset(
    {CORRIDOR_SOURCE_UNIT_KIND_V1, EXEMPLAR_SOURCE_UNIT_KIND_V1}
)
_SHA256_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")


class BehavioralLearningBatchError(ValueError):
    """A behavior source unit is malformed or is not sealed P5.4 data."""


@dataclass(frozen=True)
class BehavioralSourceUnitV1:
    """Validated finite-JSON source-unit authority for one learning batch."""

    kind: Literal["corridor_coordinate", "exemplar_passport"]
    behavioral_source_identity: str
    split_identity: str
    partition: Literal["training", "held_out"]
    coordinate: Mapping[str, object]


def corridor_source_unit_learning_batch_v1(
    materialization: BehavioralBatchesV1,
    *,
    partition: Literal["training", "held_out"],
    coordinate: tuple[str, int, int],
) -> LearningBatch:
    """Create the deterministic B=1 batch for one sealed corridor coordinate."""

    sealed = _sealed_materialization(materialization)
    example_id, position, mode_id = _corridor_coordinate(coordinate)
    batch = _corridor_partition(sealed, partition)
    row, matched_weight = _find_corridor_coordinate(
        batch, example_id=example_id, position=position, mode_id=mode_id
    )
    inputs = {
        "token_ids": (_int_row(batch.input_ids[row], "corridor token IDs"),),
        "attention_mask": (_int_row(batch.attention_mask[row], "corridor mask"),),
    }
    targets = {
        "position": (position,),
        "mode_id": (mode_id,),
        "assignment_weight": (matched_weight,),
        "mode_bounds": _mode_bounds_payload(batch),
    }
    source_coordinate = {
        "example_id": example_id,
        "position": position,
        "mode_id": mode_id,
    }
    return _build_learning_batch(
        kind=CORRIDOR_SOURCE_UNIT_KIND_V1,
        behavioral_source_identity=sealed.split.behavioral_source_identity,
        split_identity=sealed.split.split_identity,
        partition=partition,
        coordinate=source_coordinate,
        inputs=inputs,
        targets=targets,
    )


def exemplar_source_unit_learning_batch_v1(
    materialization: BehavioralBatchesV1,
    *,
    partition: Literal["training", "held_out"],
    passport_key: tuple[str, int, str],
) -> LearningBatch:
    """Create the deterministic B=1 batch for one sealed exemplar passport."""

    sealed = _sealed_materialization(materialization)
    example_id, position, fingerprint = _passport_coordinate(passport_key)
    batch = _exemplar_partition(sealed, partition)
    row, target = _find_exemplar_passport(
        batch,
        example_id=example_id,
        position=position,
        fingerprint=fingerprint,
    )
    inputs = {
        "token_ids": (_int_row(batch.input_ids[row], "exemplar token IDs"),),
        "attention_mask": (_int_row(batch.attention_mask[row], "exemplar mask"),),
    }
    targets = {
        "selected_position": (position,),
        "top_token_ids": (_int_row(target.token_ids, "sparse target token IDs"),),
        "top_probabilities": (
            _finite_row(target.probabilities, "sparse probabilities"),
        ),
        "aggregate_tail_mass": (
            _finite_float(target.aggregate_tail_mass, "tail mass"),
        ),
    }
    source_coordinate = {
        "example_id": example_id,
        "position": position,
        "corridor_fingerprint_id": fingerprint,
    }
    return _build_learning_batch(
        kind=EXEMPLAR_SOURCE_UNIT_KIND_V1,
        behavioral_source_identity=sealed.split.behavioral_source_identity,
        split_identity=sealed.split.split_identity,
        partition=partition,
        coordinate=source_coordinate,
        inputs=inputs,
        targets=targets,
    )


def validate_behavioral_source_unit_learning_batch_v1(
    batch: LearningBatch,
) -> BehavioralSourceUnitV1:
    """Fail closed unless a batch is a complete, self-bound behavior source unit."""

    if not isinstance(batch, LearningBatch):
        raise TypeError("batch must be LearningBatch")
    metadata = batch.metadata
    _exact_keys(
        metadata,
        {
            "schema_version",
            "kind",
            "behavioral_source_identity",
            "split_identity",
            "partition",
            "coordinate",
            "source_unit_identity",
        },
        "behavior source-unit metadata",
    )
    schema = metadata["schema_version"]
    kind = metadata["kind"]
    source_identity = metadata["behavioral_source_identity"]
    split_identity = metadata["split_identity"]
    partition = metadata["partition"]
    coordinate = metadata["coordinate"]
    source_unit_identity = metadata["source_unit_identity"]
    if schema != BEHAVIOR_SOURCE_UNIT_SCHEMA_V1 or kind not in _SOURCE_UNIT_KINDS:
        raise BehavioralLearningBatchError("behavior source-unit metadata is invalid")
    if not _sha256_identity(source_identity) or not _sha256_identity(split_identity):
        raise BehavioralLearningBatchError(
            "behavior source-unit identities are invalid"
        )
    if partition not in ("training", "held_out"):
        raise BehavioralLearningBatchError("behavior source-unit partition is invalid")
    if not _sha256_identity(source_unit_identity):
        raise BehavioralLearningBatchError("behavior source-unit identity is invalid")
    if not isinstance(coordinate, Mapping):
        raise BehavioralLearningBatchError("behavior source-unit coordinate is invalid")
    _validate_source_unit_values(kind, coordinate, batch.inputs, batch.targets)
    expected_identity = _source_unit_identity(
        kind=kind,
        behavioral_source_identity=source_identity,
        split_identity=split_identity,
        partition=partition,
        coordinate=coordinate,
        inputs=batch.inputs,
        targets=batch.targets,
    )
    if source_unit_identity != expected_identity or batch.batch_id != _batch_id(
        kind, expected_identity
    ):
        raise BehavioralLearningBatchError("behavior source-unit metadata is forged")
    return BehavioralSourceUnitV1(
        kind=kind,
        behavioral_source_identity=source_identity,
        split_identity=split_identity,
        partition=partition,
        coordinate=coordinate,
    )


def _sealed_materialization(value: BehavioralBatchesV1) -> BehavioralBatchesV1:
    if type(value) is not BehavioralBatchesV1:
        raise BehavioralLearningBatchError(
            "behavior source units require sealed P5.4 materialization"
        )
    try:
        # Re-run the immutable model's complete descriptor and partition checks.
        BehavioralBatchesV1(
            split=value.split,
            training_corridor=value.training_corridor,
            held_out_corridor=value.held_out_corridor,
            training_exemplars=value.training_exemplars,
            held_out_exemplars=value.held_out_exemplars,
            descriptor=value.descriptor,
        )
    except (TypeError, ValueError) as exc:
        raise BehavioralLearningBatchError(
            "behavior source units require sealed P5.4 materialization"
        ) from exc
    return value


def _corridor_partition(
    materialization: BehavioralBatchesV1, partition: str
) -> CorridorBatchV1:
    if partition == "training":
        return materialization.training_corridor
    if partition == "held_out":
        return materialization.held_out_corridor
    raise BehavioralLearningBatchError("corridor source-unit partition is invalid")


def _exemplar_partition(
    materialization: BehavioralBatchesV1, partition: str
) -> ExemplarBatchV1:
    if partition == "training":
        return materialization.training_exemplars
    if partition == "held_out":
        return materialization.held_out_exemplars
    raise BehavioralLearningBatchError("exemplar source-unit partition is invalid")


def _corridor_coordinate(value: object) -> tuple[str, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise BehavioralLearningBatchError("corridor source-unit coordinate is invalid")
    example_id, position, mode_id = value
    if (
        not isinstance(example_id, str)
        or not example_id
        or not _plain_int(position)
        or not _plain_int(mode_id)
        or position < 0
    ):
        raise BehavioralLearningBatchError("corridor source-unit coordinate is invalid")
    return example_id, position, mode_id


def _passport_coordinate(value: object) -> tuple[str, int, str]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise BehavioralLearningBatchError("exemplar source-unit passport is invalid")
    example_id, position, fingerprint = value
    if (
        not isinstance(example_id, str)
        or not example_id
        or not _plain_int(position)
        or position < 0
        or not isinstance(fingerprint, str)
        or not fingerprint
    ):
        raise BehavioralLearningBatchError("exemplar source-unit passport is invalid")
    return example_id, position, fingerprint


def _find_corridor_coordinate(
    batch: CorridorBatchV1, *, example_id: str, position: int, mode_id: int
) -> tuple[int, float]:
    _validate_corridor_batch_shape(batch)
    global_rows = tuple(sorted({int(value) for value in batch.example_indices}))
    id_by_global_row = dict(zip(global_rows, batch.example_ids, strict=True))
    matches = tuple(
        index
        for index, (global_row, observed_position, observed_mode) in enumerate(
            zip(batch.example_indices, batch.positions, batch.mode_ids, strict=True)
        )
        if (
            id_by_global_row[int(global_row)] == example_id
            and int(observed_position) == position
            and int(observed_mode) == mode_id
        )
    )
    if len(matches) != 1:
        raise BehavioralLearningBatchError("corridor source-unit coordinate is absent")
    local_row = batch.example_ids.index(example_id)
    if position >= batch.input_ids.shape[1]:
        raise BehavioralLearningBatchError("corridor source-unit coordinate is invalid")
    weight = _finite_float(batch.assignment_weights[matches[0]], "assignment weight")
    if weight < 0:
        raise BehavioralLearningBatchError("corridor source-unit weight is invalid")
    return local_row, weight


def _find_exemplar_passport(
    batch: ExemplarBatchV1,
    *,
    example_id: str,
    position: int,
    fingerprint: str,
):
    _validate_exemplar_batch_shape(batch)
    matches = tuple(
        index
        for index, passport in enumerate(batch.passports)
        if _passport_key(passport) == (example_id, position, fingerprint)
    )
    if len(matches) != 1:
        raise BehavioralLearningBatchError("exemplar source-unit passport is absent")
    row = matches[0]
    if (
        int(batch.selected_positions[row]) != position
        or position >= batch.input_ids.shape[1]
    ):
        raise BehavioralLearningBatchError("exemplar source-unit passport is invalid")
    return row, batch.sparse_targets[row]


def _mode_bounds_payload(batch: CorridorBatchV1) -> dict[str, object]:
    result: dict[str, object] = {}
    for mode_id in batch.mode_bounds.declared_mode_ids:
        statistics = batch.mode_bounds.statistics_by_mode.get(mode_id)
        if statistics is None:
            raise BehavioralLearningBatchError(
                "corridor source-unit bounds are invalid"
            )
        result[str(mode_id)] = {
            name: (
                _finite_float(bound.minimum, "mode lower bound"),
                _finite_float(bound.mean, "mode mean bound"),
                _finite_float(bound.maximum, "mode upper bound"),
            )
            for name, bound in sorted(statistics.items())
        }
    if not result:
        raise BehavioralLearningBatchError("corridor source-unit bounds are invalid")
    return result


def _build_learning_batch(
    *,
    kind: str,
    behavioral_source_identity: str,
    split_identity: str,
    partition: str,
    coordinate: Mapping[str, object],
    inputs: Mapping[str, object],
    targets: Mapping[str, object],
) -> LearningBatch:
    identity = _source_unit_identity(
        kind=kind,
        behavioral_source_identity=behavioral_source_identity,
        split_identity=split_identity,
        partition=partition,
        coordinate=coordinate,
        inputs=inputs,
        targets=targets,
    )
    return LearningBatch(
        batch_id=_batch_id(kind, identity),
        inputs=inputs,
        targets=targets,
        metadata={
            "schema_version": BEHAVIOR_SOURCE_UNIT_SCHEMA_V1,
            "kind": kind,
            "behavioral_source_identity": behavioral_source_identity,
            "split_identity": split_identity,
            "partition": partition,
            "coordinate": coordinate,
            "source_unit_identity": identity,
        },
    )


def _source_unit_identity(
    *,
    kind: object,
    behavioral_source_identity: object,
    split_identity: object,
    partition: object,
    coordinate: object,
    inputs: object,
    targets: object,
) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                json_value(
                    {
                        "schema_version": BEHAVIOR_SOURCE_UNIT_SCHEMA_V1,
                        "kind": kind,
                        "behavioral_source_identity": behavioral_source_identity,
                        "split_identity": split_identity,
                        "partition": partition,
                        "coordinate": coordinate,
                        "inputs": inputs,
                        "targets": targets,
                    }
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )


def _batch_id(kind: object, source_unit_identity: object) -> str:
    return f"behavior_source_unit.v1:{kind}:{source_unit_identity}"


def _validate_source_unit_values(
    kind: str,
    coordinate: Mapping[str, object],
    inputs: Mapping[str, object],
    targets: Mapping[str, object],
) -> None:
    _exact_keys(inputs, {"token_ids", "attention_mask"}, "behavior source-unit inputs")
    token_ids = _one_int_row(inputs["token_ids"], "behavior source-unit token IDs")
    mask = _one_int_row(inputs["attention_mask"], "behavior source-unit mask")
    if len(token_ids) != len(mask) or any(value not in (0, 1) for value in mask):
        raise BehavioralLearningBatchError("behavior source-unit inputs are invalid")
    if kind == CORRIDOR_SOURCE_UNIT_KIND_V1:
        _validate_corridor_source_unit(coordinate, targets, len(token_ids))
    elif kind == EXEMPLAR_SOURCE_UNIT_KIND_V1:
        _validate_exemplar_source_unit(coordinate, targets, len(token_ids))
    else:  # pragma: no cover - caller validates the kind first.
        raise BehavioralLearningBatchError("behavior source-unit kind is invalid")


def _validate_corridor_source_unit(
    coordinate: Mapping[str, object], targets: Mapping[str, object], length: int
) -> None:
    _exact_keys(
        coordinate,
        {"example_id", "position", "mode_id"},
        "corridor source-unit coordinate",
    )
    example_id, position, mode_id = (
        coordinate["example_id"],
        coordinate["position"],
        coordinate["mode_id"],
    )
    if (
        not isinstance(example_id, str)
        or not example_id
        or not _plain_int(position)
        or not _plain_int(mode_id)
        or not 0 <= position < length
    ):
        raise BehavioralLearningBatchError("corridor source-unit coordinate is invalid")
    _exact_keys(
        targets,
        {"position", "mode_id", "assignment_weight", "mode_bounds"},
        "corridor source-unit targets",
    )
    if (
        _one_int_value(targets["position"], "corridor position") != position
        or _one_int_value(targets["mode_id"], "corridor mode") != mode_id
    ):
        raise BehavioralLearningBatchError("corridor source-unit targets are forged")
    weight = _one_finite_value(targets["assignment_weight"], "assignment weight")
    if weight < 0:
        raise BehavioralLearningBatchError("corridor source-unit targets are invalid")
    _validate_mode_bounds_payload(targets["mode_bounds"])


def _validate_exemplar_source_unit(
    coordinate: Mapping[str, object], targets: Mapping[str, object], length: int
) -> None:
    _exact_keys(
        coordinate,
        {"example_id", "position", "corridor_fingerprint_id"},
        "exemplar source-unit coordinate",
    )
    example_id, position, fingerprint = (
        coordinate["example_id"],
        coordinate["position"],
        coordinate["corridor_fingerprint_id"],
    )
    if (
        not isinstance(example_id, str)
        or not example_id
        or not _plain_int(position)
        or not 0 <= position < length
        or not isinstance(fingerprint, str)
        or not fingerprint
    ):
        raise BehavioralLearningBatchError("exemplar source-unit coordinate is invalid")
    _exact_keys(
        targets,
        {
            "selected_position",
            "top_token_ids",
            "top_probabilities",
            "aggregate_tail_mass",
        },
        "exemplar source-unit targets",
    )
    if _one_int_value(targets["selected_position"], "selected position") != position:
        raise BehavioralLearningBatchError("exemplar source-unit targets are forged")
    token_ids = _one_int_row(targets["top_token_ids"], "sparse target token IDs")
    probabilities = _one_finite_row(
        targets["top_probabilities"], "sparse target probabilities"
    )
    tail_mass = _one_finite_value(targets["aggregate_tail_mass"], "tail mass")
    if (
        not token_ids
        or len(token_ids) != len(probabilities)
        or any(value < 0 for value in token_ids)
        or any(value < 0 for value in probabilities)
        or tail_mass < 0
        or not np.isclose(sum(probabilities) + tail_mass, 1.0)
    ):
        raise BehavioralLearningBatchError("exemplar source-unit targets are invalid")


def _validate_mode_bounds_payload(value: object) -> None:
    if not isinstance(value, Mapping) or not value:
        raise BehavioralLearningBatchError("corridor source-unit bounds are invalid")
    for mode_id, statistics in value.items():
        try:
            int(mode_id)
        except (TypeError, ValueError) as exc:
            raise BehavioralLearningBatchError(
                "corridor source-unit bounds are invalid"
            ) from exc
        if not isinstance(statistics, Mapping) or not statistics:
            raise BehavioralLearningBatchError(
                "corridor source-unit bounds are invalid"
            )
        for interval in statistics.values():
            values = _finite_row(interval, "corridor mode interval")
            if len(values) != 3 or values[0] > values[1] or values[1] > values[2]:
                raise BehavioralLearningBatchError(
                    "corridor source-unit bounds are invalid"
                )


def _validate_corridor_batch_shape(batch: CorridorBatchV1) -> None:
    if (
        batch.partition not in ("training", "held_out")
        or batch.input_ids.ndim != 2
        or batch.attention_mask.shape != batch.input_ids.shape
        or batch.input_ids.shape[0] != len(batch.example_ids)
        or len(set(batch.example_ids)) != len(batch.example_ids)
        or any(not isinstance(value, str) or not value for value in batch.example_ids)
        or len(batch.example_indices) != len(batch.positions)
        or len(batch.positions) != len(batch.mode_ids)
        or len(batch.mode_ids) != len(batch.assignment_weights)
    ):
        raise BehavioralLearningBatchError("sealed corridor materialization is invalid")
    global_rows = {int(value) for value in batch.example_indices}
    if len(global_rows) != len(batch.example_ids):
        raise BehavioralLearningBatchError("sealed corridor materialization is invalid")


def _validate_exemplar_batch_shape(batch: ExemplarBatchV1) -> None:
    if (
        batch.partition not in ("training", "held_out")
        or batch.input_ids.ndim != 2
        or batch.attention_mask.shape != batch.input_ids.shape
        or batch.input_ids.shape[0] != len(batch.example_ids)
        or len(batch.example_ids) != len(batch.passports)
        or len(batch.passports) != len(batch.sparse_targets)
        or len(batch.selected_positions) != len(batch.passports)
        or len(batch.selected_example_indices) != len(batch.passports)
    ):
        raise BehavioralLearningBatchError("sealed exemplar materialization is invalid")


def _passport_key(value: Mapping[str, object]) -> tuple[str, int, str]:
    try:
        example_id = value["selected_example_id"]
        position = value["selected_position"]
        fingerprint = value["corridor_fingerprint_id"]
    except (KeyError, TypeError) as exc:
        raise BehavioralLearningBatchError(
            "sealed exemplar passport is invalid"
        ) from exc
    if (
        not isinstance(example_id, str)
        or not example_id
        or not _plain_int(position)
        or not isinstance(fingerprint, str)
        or not fingerprint
    ):
        raise BehavioralLearningBatchError("sealed exemplar passport is invalid")
    return example_id, position, fingerprint


def _int_row(value: object, name: str) -> tuple[int, ...]:
    values = tuple(np.asarray(value).tolist())
    if not values or any(not _plain_int(item) or item < 0 for item in values):
        raise BehavioralLearningBatchError(f"{name} must be nonnegative integers")
    return values


def _finite_row(value: object, name: str) -> tuple[float, ...]:
    values = tuple(np.asarray(value).tolist())
    if not values:
        raise BehavioralLearningBatchError(f"{name} must be nonempty")
    return tuple(_finite_float(item, name) for item in values)


def _one_int_row(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise BehavioralLearningBatchError(f"{name} must have B=1")
    return _int_row(value[0], name)


def _one_finite_row(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise BehavioralLearningBatchError(f"{name} must have B=1")
    return _finite_row(value[0], name)


def _one_int_value(value: object, name: str) -> int:
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise BehavioralLearningBatchError(f"{name} must contain one value")
    item = value[0]
    if not _plain_int(item) or item < 0:
        raise BehavioralLearningBatchError(f"{name} must be a nonnegative integer")
    return int(item)


def _one_finite_value(value: object, name: str) -> float:
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise BehavioralLearningBatchError(f"{name} must contain one value")
    return _finite_float(value[0], name)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise BehavioralLearningBatchError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise BehavioralLearningBatchError(f"{name} must be finite")
    return result


def _plain_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise BehavioralLearningBatchError(f"{name} has unsupported fields")


def _sha256_identity(value: object) -> bool:
    return isinstance(value, str) and _SHA256_IDENTITY.fullmatch(value) is not None


__all__ = [
    "BEHAVIOR_SOURCE_UNIT_SCHEMA_V1",
    "CORRIDOR_SOURCE_UNIT_KIND_V1",
    "EXEMPLAR_SOURCE_UNIT_KIND_V1",
    "BehavioralLearningBatchError",
    "BehavioralSourceUnitV1",
    "corridor_source_unit_learning_batch_v1",
    "exemplar_source_unit_learning_batch_v1",
    "validate_behavioral_source_unit_learning_batch_v1",
]
