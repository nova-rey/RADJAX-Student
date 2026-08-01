"""Typed, JAX-free extraction from a Contract-admitted native-v3 artifact.

Contract validates artifact semantics and opens each declared resource with a
fresh raw-integrity check.  This module only decodes those verified streams
into immutable Student views; it neither discovers files nor defines batches,
losses, or learning policy.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from radjax_contract.tome import open_verified_student_resource_v4

from radjax_student.artifacts.native_v3 import NativeV3StudentConsumptionView


@dataclass(frozen=True)
class NativeV3VerifiedPayloadResource:
    resource_id: str
    role: str
    instance_id: str
    semantic_digest: str
    raw_sha256: str
    raw_size_bytes: int
    encoding: str
    classification: str
    consumption: Mapping[str, Any]


@dataclass(frozen=True)
class NativeV3TargetShard:
    resource: NativeV3VerifiedPayloadResource
    input_ids: np.ndarray
    attention_mask: np.ndarray
    corridor_lengths: np.ndarray


@dataclass(frozen=True)
class NativeV3CorridorAssignments:
    resource: NativeV3VerifiedPayloadResource
    example_indices: np.ndarray
    positions: np.ndarray
    mode_ids: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class NativeV3ObservedCorridorStatistics:
    resource: NativeV3VerifiedPayloadResource
    entropy: np.ndarray
    top1_margin: np.ndarray
    top8_mass: np.ndarray
    top32_mass: np.ndarray
    tail_mass: np.ndarray


@dataclass(frozen=True)
class NativeV3StudentPayloadView:
    """Verified decoded resources; intentionally not a learning batch."""

    consumption: NativeV3StudentConsumptionView
    target_shard: NativeV3TargetShard
    example_registry: tuple[Mapping[str, Any], ...]
    corridor_mode_table: Mapping[str, Any]
    corridor_assignments: NativeV3CorridorAssignments
    selected_passports: tuple[Mapping[str, Any], ...]
    selected_exemplar_payloads: tuple[Mapping[str, Any], ...]
    observed_statistics: NativeV3ObservedCorridorStatistics
    row_range_declaration: Mapping[str, Any]
    delivery: Mapping[str, Any]
    provenance: Mapping[str, Any]


class NativeV3PayloadError(ValueError):
    """A declared resource could not be decoded after Contract admission."""


def load_native_v3_student_payloads(
    consumption: NativeV3StudentConsumptionView, *, strict: bool = False
) -> NativeV3StudentPayloadView:
    """Decode every Contract-declared batch/validation resource exactly once."""

    descriptor = consumption.descriptor
    resources = _resources_by_role(descriptor)
    target = _decode_target_shard(consumption, resources["target_shard"], strict)
    assignments = _decode_assignments(
        consumption, resources["corridor_assignment"], strict
    )
    observed = _decode_observed_statistics(
        consumption, resources["corridor_observed_statistics"], strict
    )
    return NativeV3StudentPayloadView(
        consumption=consumption,
        target_shard=target,
        example_registry=_records(
            _open_json(consumption, resources["example_registry"], strict),
            "examples",
            "example_registry",
        ),
        corridor_mode_table=_frozen_mapping(
            _open_json(consumption, resources["corridor_mode_table"], strict),
            "corridor_mode_table",
        ),
        corridor_assignments=assignments,
        selected_passports=_records(
            _open_json(consumption, resources["selected_passport_index"], strict),
            "selected_exemplars",
            "selected_passport_index",
        ),
        selected_exemplar_payloads=_records(
            _open_json(consumption, resources["selected_exemplar_payload"], strict),
            "selected_exemplars",
            "selected_exemplar_payload",
        ),
        observed_statistics=observed,
        row_range_declaration=_frozen_mapping(
            _open_json(consumption, resources["row_range_declaration"], strict),
            "row_range_declaration",
        ),
        delivery=_freeze(descriptor.delivery),
        provenance=_freeze(descriptor.provenance),
    )


def _resources_by_role(descriptor: Any) -> dict[str, Any]:
    resources = (
        *descriptor.corridor_resources,
        *descriptor.exemplar_resources,
        *descriptor.validation_resources,
    )
    by_role = {resource.role: resource for resource in resources}
    required = {
        "target_shard",
        "example_registry",
        "corridor_mode_table",
        "corridor_assignment",
        "selected_passport_index",
        "selected_exemplar_payload",
        "corridor_observed_statistics",
        "row_range_declaration",
        "delivery_receipt",
        "authority_reference",
    }
    if set(by_role) != required or len(by_role) != len(resources):
        raise NativeV3PayloadError("Contract descriptor has unexpected resource roles")
    return by_role


def _resource_metadata(resource: Any) -> NativeV3VerifiedPayloadResource:
    return NativeV3VerifiedPayloadResource(
        resource_id=resource.resource_id,
        role=resource.role,
        instance_id=resource.instance_id,
        semantic_digest=resource.semantic_digest,
        raw_sha256=resource.raw_sha256,
        raw_size_bytes=resource.raw_size_bytes,
        encoding=resource.encoding,
        classification=resource.classification,
        consumption=_freeze(resource.consumption),
    )


def _open_bytes(
    consumption: NativeV3StudentConsumptionView, resource: Any, strict: bool
) -> bytes:
    with open_verified_student_resource_v4(
        consumption.artifact_path, resource.resource_id, strict=strict
    ) as handle:
        return handle.read()


def _open_json(
    consumption: NativeV3StudentConsumptionView, resource: Any, strict: bool
) -> Mapping[str, Any]:
    if resource.encoding != "json":
        raise NativeV3PayloadError(f"{resource.role} must be declared as JSON")
    try:
        payload = json.loads(_open_bytes(consumption, resource, strict))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeV3PayloadError(
            f"invalid verified JSON resource: {resource.role}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeV3PayloadError(f"JSON resource must be an object: {resource.role}")
    return payload


def _open_npz(
    consumption: NativeV3StudentConsumptionView, resource: Any, strict: bool
) -> Mapping[str, np.ndarray]:
    if resource.encoding != "npz":
        raise NativeV3PayloadError(f"{resource.role} must be declared as NPZ")
    try:
        with np.load(
            io.BytesIO(_open_bytes(consumption, resource, strict)), allow_pickle=False
        ) as archive:
            return {
                name: _readonly(np.array(archive[name], copy=True))
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise NativeV3PayloadError(
            f"invalid verified NPZ resource: {resource.role}"
        ) from exc


def _decode_target_shard(
    consumption: NativeV3StudentConsumptionView, resource: Any, strict: bool
) -> NativeV3TargetShard:
    payload = _open_npz(consumption, resource, strict)
    _require_keys(
        payload, {"input_ids", "attention_mask", "corridor_lengths"}, resource.role
    )
    return NativeV3TargetShard(
        resource=_resource_metadata(resource),
        input_ids=payload["input_ids"],
        attention_mask=payload["attention_mask"],
        corridor_lengths=payload["corridor_lengths"],
    )


def _decode_assignments(
    consumption: NativeV3StudentConsumptionView, resource: Any, strict: bool
) -> NativeV3CorridorAssignments:
    payload = _open_npz(consumption, resource, strict)
    _require_keys(
        payload,
        {"position_example_index", "position", "mode_id", "weight"},
        resource.role,
    )
    return NativeV3CorridorAssignments(
        resource=_resource_metadata(resource),
        example_indices=payload["position_example_index"],
        positions=payload["position"],
        mode_ids=payload["mode_id"],
        weights=payload["weight"],
    )


def _decode_observed_statistics(
    consumption: NativeV3StudentConsumptionView, resource: Any, strict: bool
) -> NativeV3ObservedCorridorStatistics:
    payload = _open_npz(consumption, resource, strict)
    fields = {"entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass"}
    _require_keys(payload, fields, resource.role)
    return NativeV3ObservedCorridorStatistics(
        resource=_resource_metadata(resource),
        entropy=payload["entropy"],
        top1_margin=payload["top1_margin"],
        top8_mass=payload["top8_mass"],
        top32_mass=payload["top32_mass"],
        tail_mass=payload["tail_mass"],
    )


def _require_keys(payload: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(payload) != expected:
        raise NativeV3PayloadError(f"unexpected verified NPZ members for {role}")


def _records(
    payload: Mapping[str, Any], key: str, role: str
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise NativeV3PayloadError(f"invalid verified record envelope for {role}")
    return tuple(_freeze(item) for item in value)


def _frozen_mapping(value: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeV3PayloadError(f"invalid verified mapping for {role}")
    return _freeze(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value
