"""Materialize P5.3 immutable authority into neutral P5.4 batch values."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from radjax_student.artifacts.native_v3_v6 import NativeV3V6BehavioralProjection
from radjax_student.behavior.models import (
    BehavioralBatchesV1,
    CorridorBatchV1,
    ExemplarBatchV1,
    ModeBoundsV1,
    ModeStatisticBoundsV1,
    SparseTargetV1,
    frozen_mapping,
)
from radjax_student.behavior.policies import BehaviorSplitError, BehaviorSplitPolicyV1


class BehavioralMaterializationError(ValueError):
    """P5.3 values cannot safely become neutral P5.4 batches."""


def materialize_behavioral_batches_v1(
    projection: NativeV3V6BehavioralProjection,
    *,
    split_policy: BehaviorSplitPolicyV1 | None = None,
) -> BehavioralBatchesV1:
    """Terminate artifact authority at a fully validated neutral batch boundary."""

    policy = split_policy or BehaviorSplitPolicyV1()
    example_ids = _example_ids(projection.example_registry)
    target_ids, target_mask = _target_tensors(projection, len(example_ids))
    assignments = _assignment_tensors(projection, len(example_ids), target_ids.shape[1])
    passports, payloads = _join_exemplars(
        projection.selected_passports,
        projection.selected_exemplar_payloads,
        example_ids,
    )
    try:
        split = policy.split(
            behavioral_source_identity=projection.behavioral_source_identity,
            example_ids=example_ids,
            exemplar_example_ids=tuple(
                record["selected_example_id"] for record in passports
            ),
        )
    except BehaviorSplitError as exc:
        raise BehavioralMaterializationError("behavioral split is invalid") from exc
    bounds = _mode_bounds(projection.corridor_mode_table.content, assignments[2])
    return BehavioralBatchesV1(
        split=split,
        training_corridor=_corridor_batch(
            "training",
            split.assignments,
            example_ids,
            target_ids,
            target_mask,
            assignments,
            bounds,
        ),
        held_out_corridor=_corridor_batch(
            "held_out",
            split.assignments,
            example_ids,
            target_ids,
            target_mask,
            assignments,
            bounds,
        ),
        training_exemplars=_exemplar_batch(
            "training",
            split.assignments,
            example_ids,
            target_ids,
            target_mask,
            passports,
            payloads,
        ),
        held_out_exemplars=_exemplar_batch(
            "held_out",
            split.assignments,
            example_ids,
            target_ids,
            target_mask,
            passports,
            payloads,
        ),
    )


def _example_ids(records: tuple[Mapping[str, object], ...]) -> tuple[str, ...]:
    values = tuple(record.get("example_id") for record in records)
    if any(not isinstance(value, str) or not value for value in values) or len(
        set(values)
    ) != len(values):
        raise BehavioralMaterializationError("example registry IDs are invalid")
    return values  # Contract registry order is its coordinate order.


def _target_tensors(
    projection: NativeV3V6BehavioralProjection, count: int
) -> tuple[np.ndarray, np.ndarray]:
    components = projection.target_shard.components
    if set(components) != {"input_ids", "attention_mask"}:
        raise BehavioralMaterializationError("target shard components are invalid")
    input_ids = _readonly(components["input_ids"].values)
    mask = _readonly(components["attention_mask"].values)
    if (
        input_ids.ndim != 2
        or input_ids.shape != mask.shape
        or input_ids.shape[0] != count
    ):
        raise BehavioralMaterializationError("target shard shapes are invalid")
    if input_ids.dtype.hasobject or mask.dtype.hasobject:
        raise BehavioralMaterializationError("target shard dtype is invalid")
    return input_ids, mask


def _assignment_tensors(
    projection: NativeV3V6BehavioralProjection, count: int, length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    components = projection.corridor_assignment.components
    required = {"example_index", "position", "mode_id", "weight"}
    if set(components) != required:
        raise BehavioralMaterializationError(
            "corridor assignment components are invalid"
        )
    values = tuple(
        _readonly(components[name].values)
        for name in ("example_index", "position", "mode_id", "weight")
    )
    if (
        any(value.ndim != 1 for value in values)
        or len({value.shape[0] for value in values}) != 1
    ):
        raise BehavioralMaterializationError("corridor assignment shapes are invalid")
    example_index, position, mode_id, weight = values
    if (
        not np.issubdtype(example_index.dtype, np.integer)
        or not np.issubdtype(position.dtype, np.integer)
        or not np.issubdtype(mode_id.dtype, np.integer)
        or not np.issubdtype(weight.dtype, np.floating)
    ):
        raise BehavioralMaterializationError("corridor assignment dtypes are invalid")
    if (
        np.any(example_index < 0)
        or np.any(example_index >= count)
        or np.any(position < 0)
        or np.any(position >= length)
        or not np.all(np.isfinite(weight))
        or np.any(weight < 0)
    ):
        raise BehavioralMaterializationError(
            "corridor assignment coordinates are invalid"
        )
    return example_index, position, mode_id, weight


def _join_exemplars(
    passports: tuple[Mapping[str, object], ...],
    payloads: tuple[Mapping[str, object], ...],
    example_ids: tuple[str, ...],
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    def key(item: Mapping[str, object]) -> tuple[object, object, object]:
        return (
            item.get("selected_example_id"),
            item.get("selected_position"),
            item.get("corridor_fingerprint_id"),
        )

    if not passports or len(passports) != len(payloads):
        raise BehavioralMaterializationError("selected exemplar records are incomplete")
    passport_by_key = {key(item): item for item in passports}
    payload_by_key = {key(item): item for item in payloads}
    if len(passport_by_key) != len(passports) or set(passport_by_key) != set(
        payload_by_key
    ):
        raise BehavioralMaterializationError("selected exemplar records do not join")
    ordered = tuple(
        sorted(
            passport_by_key,
            key=lambda value: (
                str(value[0]).encode("utf-8"),
                value[1],
                str(value[2]).encode("utf-8"),
            ),
        )
    )
    result_passports = tuple(frozen_mapping(passport_by_key[item]) for item in ordered)
    result_payloads = tuple(frozen_mapping(payload_by_key[item]) for item in ordered)
    if any(
        item.get("selected_example_id") not in example_ids
        or type(item.get("selected_position")) is not int
        for item in result_passports
    ):
        raise BehavioralMaterializationError("selected exemplar coordinate is invalid")
    return result_passports, result_payloads


def _mode_bounds(table: Mapping[str, object], mode_ids: np.ndarray) -> ModeBoundsV1:
    modes = table.get("modes")
    if not isinstance(modes, tuple) or not modes:
        raise BehavioralMaterializationError("corridor mode table is invalid")
    if not all(isinstance(item, Mapping) for item in modes):
        raise BehavioralMaterializationError("corridor mode table is invalid")
    declarations = tuple(frozen_mapping(item) for item in modes)
    declared = tuple(item.get("mode_id") for item in declarations)
    if (
        len(set(declared)) != len(declared)
        or any(type(value) is not int for value in declared)
        or not set(mode_ids.tolist()).issubset(set(declared))
    ):
        raise BehavioralMaterializationError("corridor mode IDs are invalid")
    ordered = tuple(sorted(declared))
    return ModeBoundsV1(
        minimum_mode_id=ordered[0],
        maximum_mode_id=ordered[-1],
        declared_mode_ids=ordered,
        declarations=declarations,
        statistics_by_mode=_mode_statistics(declarations),
    )


_MODE_STATISTICS = frozenset(
    {"entropy", "tail_mass", "top1_margin", "top8_mass", "top32_mass"}
)


def _mode_statistics(
    declarations: tuple[Mapping[str, object], ...],
) -> Mapping[int, Mapping[str, ModeStatisticBoundsV1]]:
    result: dict[int, Mapping[str, ModeStatisticBoundsV1]] = {}
    for declaration in declarations:
        mode_id = declaration["mode_id"]
        statistics = declaration.get("statistics")
        if not isinstance(statistics, Mapping) or set(statistics) != _MODE_STATISTICS:
            raise BehavioralMaterializationError("corridor mode statistics are invalid")
        intervals: dict[str, ModeStatisticBoundsV1] = {}
        for statistic, interval in statistics.items():
            if not isinstance(interval, Mapping) or set(interval) != {
                "min",
                "mean",
                "max",
            }:
                raise BehavioralMaterializationError(
                    "corridor mode statistic interval is invalid"
                )
            minimum, mean, maximum = (
                interval["min"],
                interval["mean"],
                interval["max"],
            )
            if (
                any(
                    not isinstance(value, (int, float))
                    for value in (minimum, mean, maximum)
                )
                or not all(np.isfinite(value) for value in (minimum, mean, maximum))
                or minimum > mean
                or mean > maximum
            ):
                raise BehavioralMaterializationError(
                    "corridor mode statistic interval is invalid"
                )
            intervals[statistic] = ModeStatisticBoundsV1(
                minimum=float(minimum), mean=float(mean), maximum=float(maximum)
            )
        result[mode_id] = MappingProxyType(dict(sorted(intervals.items())))
    return MappingProxyType(dict(sorted(result.items())))


def _corridor_batch(
    partition: str,
    assignments: Mapping[str, str],
    example_ids: tuple[str, ...],
    input_ids: np.ndarray,
    mask: np.ndarray,
    tensors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bounds: ModeBoundsV1,
) -> CorridorBatchV1:
    selected = tuple(
        index
        for index, example_id in enumerate(example_ids)
        if assignments[example_id] == partition
    )
    if not selected:
        raise BehavioralMaterializationError("corridor partition is empty")
    example_index, position, mode_id, weight = tensors
    include = np.isin(example_index, np.asarray(selected, dtype=example_index.dtype))
    if not np.any(include):
        raise BehavioralMaterializationError("corridor partition has no assignments")
    return CorridorBatchV1(
        partition=partition,
        example_ids=tuple(example_ids[index] for index in selected),
        input_ids=_readonly(input_ids[list(selected)]),
        attention_mask=_readonly(mask[list(selected)]),
        example_indices=_readonly(example_index[include]),
        positions=_readonly(position[include]),
        mode_ids=_readonly(mode_id[include]),
        mode_bounds=bounds,
        assignment_weights=_readonly(weight[include]),
    )


def _exemplar_batch(
    partition: str,
    assignments: Mapping[str, str],
    example_ids: tuple[str, ...],
    input_ids: np.ndarray,
    mask: np.ndarray,
    passports: tuple[Mapping[str, object], ...],
    payloads: tuple[Mapping[str, object], ...],
) -> ExemplarBatchV1:
    selected = tuple(
        index
        for index, passport in enumerate(passports)
        if assignments[str(passport["selected_example_id"])] == partition
    )
    if not selected:
        raise BehavioralMaterializationError("exemplar partition is empty")
    coordinates = [
        (
            example_ids.index(str(passports[index]["selected_example_id"])),
            passports[index]["selected_position"],
        )
        for index in selected
    ]
    if any(
        position < 0 or position >= input_ids.shape[1] for _, position in coordinates
    ):
        raise BehavioralMaterializationError("selected exemplar position is invalid")
    sparse = tuple(_sparse_target(payloads[index]) for index in selected)
    return ExemplarBatchV1(
        partition=partition,
        example_ids=tuple(
            str(passports[index]["selected_example_id"]) for index in selected
        ),
        input_ids=_readonly(np.asarray([input_ids[row] for row, _ in coordinates])),
        attention_mask=_readonly(np.asarray([mask[row] for row, _ in coordinates])),
        selected_example_indices=_readonly(
            np.asarray([row for row, _ in coordinates], dtype=np.int32)
        ),
        selected_positions=_readonly(
            np.asarray([position for _, position in coordinates], dtype=np.int32)
        ),
        sparse_targets=sparse,
        passports=tuple(passports[index] for index in selected),
    )


def _sparse_target(payload: Mapping[str, object]) -> SparseTargetV1:
    token_ids, probabilities, tail = (
        payload.get("top_token_ids"),
        payload.get("top_probs"),
        payload.get("tail_mass"),
    )
    if (
        not isinstance(token_ids, tuple)
        or not isinstance(probabilities, tuple)
        or len(token_ids) == 0
        or len(token_ids) != len(probabilities)
        or any(type(value) is not int or value < 0 for value in token_ids)
        or any(
            not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
            for value in probabilities
        )
        or not isinstance(tail, (int, float))
        or not np.isfinite(tail)
        or tail < 0
    ):
        raise BehavioralMaterializationError("selected sparse target is invalid")
    return SparseTargetV1(
        token_ids=_readonly(np.asarray(token_ids, dtype=np.int32)),
        probabilities=_readonly(np.asarray(probabilities, dtype=np.float32)),
        aggregate_tail_mass=float(tail),
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result
