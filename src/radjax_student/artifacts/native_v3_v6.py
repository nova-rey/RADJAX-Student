"""Strict, transport-neutral projection of Contract v6 behavioral authority.

Contract owns admission, archive transport, resource integrity, and behavioral
semantics.  Student retains only the identity-bearing projection required by
later neutral policy checkpoints.  In particular, no locator, archive member,
or temporary path is retained here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from radjax_contract.tome import (
    open_verified_student_jsonl_records_v6,
    open_verified_student_m7_payload_v6,
    open_verified_student_multipart_resource_v6,
    open_verified_student_resource_v6,
    resolve_student_language_binding,
    validate_and_resolve_student_consumption,
)

from radjax_student.hf.language_projection import (
    HFLanguageProjectionV1,
    project_hf_language_binding,
)

NATIVE_V3_STUDENT_V6_PROFILE = "native_v3_student_v6"
_AUTHORITY_RESOURCE_IDS = {
    "authority_reference": "authority_reference/default",
    "corridor_assignment": "corridor_assignment/default",
    "corridor_mode_table": "corridor_mode_table/default",
    "example_registry": "example_registry/default",
    "selected_exemplar_payload": "selected_exemplar_payload/default",
    "selected_passport_index": "selected_passport_index/default",
    "target_shard": "target_shard/default",
}


class NativeV3V6ProjectionError(ValueError):
    """The public Contract projection could not become a Student value."""


@dataclass(frozen=True)
class NativeV3V6Component:
    """One immutable NPY component with its Contract-declared identities."""

    component_id: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    semantic_identity: str
    raw_sha256: str
    raw_size_bytes: int
    values: np.ndarray


@dataclass(frozen=True)
class NativeV3V6MultipartResource:
    """A complete semantic multipart resource, never a physical member view."""

    resource_id: str
    role: str
    schema: str
    semantic_identity: str
    components: Mapping[str, NativeV3V6Component]


@dataclass(frozen=True)
class NativeV3V6JsonResource:
    """One whole verified JSON authority resource without a physical locator."""

    resource_id: str
    role: str
    schema: str
    semantic_identity: str
    raw_sha256: str
    raw_size_bytes: int
    content: Mapping[str, Any]


@dataclass(frozen=True)
class NativeV3V6BehavioralProjection:
    """The complete P5.3 authority projection; intentionally not a batch."""

    language: HFLanguageProjectionV1
    behavioral_source_identity: str
    behavioral_authority_digest: str
    package_semantic_identity: str
    composition_digest: str
    authority_reference: NativeV3V6JsonResource
    corridor_mode_table: NativeV3V6JsonResource
    target_shard: NativeV3V6MultipartResource
    corridor_assignment: NativeV3V6MultipartResource
    example_registry: tuple[Mapping[str, Any], ...]
    selected_passports: tuple[Mapping[str, Any], ...]
    selected_exemplar_payloads: tuple[Mapping[str, Any], ...]


def open_native_v3_v6_behavioral_projection(
    artifact: Any, *, strict: bool = True
) -> NativeV3V6BehavioralProjection:
    """Admit and project every public v6 behavioral authority resource.

    ``strict`` exists only to make the fail-closed boundary explicit.  The
    Contract APIs are always called with strict v6 admission; permissive
    callers are rejected before any resource can be projected.
    """

    if not strict:
        raise NativeV3V6ProjectionError("v6 Student projection requires strict=True")
    result = validate_and_resolve_student_consumption(
        artifact, profile_id=NATIVE_V3_STUDENT_V6_PROFILE, strict=True
    )
    if not result.ok or result.descriptor is None:
        raise NativeV3V6ProjectionError(_admission_message(result))
    descriptor = result.descriptor
    _assert_authority_closure(descriptor)
    language_descriptor = resolve_student_language_binding(
        artifact, profile_id=NATIVE_V3_STUDENT_V6_PROFILE, strict=True
    )
    if (
        language_descriptor.canonical_binding_digest
        != descriptor.language_binding_digest
    ):
        raise NativeV3V6ProjectionError("v6 language binding digest mismatch")
    return NativeV3V6BehavioralProjection(
        language=_language_projection(language_descriptor),
        behavioral_source_identity=descriptor.behavioral_source_identity,
        behavioral_authority_digest=descriptor.behavioral_authority_digest,
        package_semantic_identity=descriptor.package_semantic_identity,
        composition_digest=descriptor.composition_digest,
        authority_reference=_json_resource_projection(
            artifact, descriptor, "authority_reference/default"
        ),
        corridor_mode_table=_json_resource_projection(
            artifact, descriptor, "corridor_mode_table/default"
        ),
        target_shard=_multipart_projection(artifact, "target_shard/default"),
        corridor_assignment=_multipart_projection(
            artifact, "corridor_assignment/default"
        ),
        example_registry=_jsonl_records(artifact, "example_registry/default"),
        selected_passports=_jsonl_records(artifact, "selected_passport_index/default"),
        selected_exemplar_payloads=_m7_records(
            artifact, "selected_exemplar_payload/default"
        ),
    )


def _admission_message(result: Any) -> str:
    codes = tuple(item.code for item in getattr(result, "issues", ()))
    return "v6 behavioral admission failed: " + ",".join(codes or ("unknown",))


def _assert_authority_closure(descriptor: Any) -> None:
    resources = tuple(getattr(descriptor, "authority_resources", ()))
    by_role = {resource.role: resource for resource in resources}
    if len(by_role) != len(resources) or set(by_role) != set(_AUTHORITY_RESOURCE_IDS):
        raise NativeV3V6ProjectionError("v6 authority resource closure is invalid")
    if any(
        by_role[role].resource_id != resource_id
        for role, resource_id in _AUTHORITY_RESOURCE_IDS.items()
    ):
        raise NativeV3V6ProjectionError("v6 authority resource closure is invalid")


def _language_projection(descriptor: Any) -> HFLanguageProjectionV1:
    try:
        return project_hf_language_binding(descriptor)
    except (TypeError, ValueError) as exc:
        raise NativeV3V6ProjectionError("v6 HF language projection is invalid") from exc


def _multipart_projection(
    artifact: Any, resource_id: str
) -> NativeV3V6MultipartResource:
    with open_verified_student_multipart_resource_v6(
        artifact, resource_id, profile_id=NATIVE_V3_STUDENT_V6_PROFILE, strict=True
    ) as resource:
        components: dict[str, NativeV3V6Component] = {}
        for component_id, component in resource.components.items():
            values = _load_component(component, resource_id)
            if component_id != component.component_id or component_id in components:
                raise NativeV3V6ProjectionError(
                    "v6 multipart component identity is invalid"
                )
            components[component_id] = NativeV3V6Component(
                component_id=component.component_id,
                dtype=component.dtype,
                shape=tuple(component.shape),
                axes=tuple(component.axes),
                semantic_identity=component.semantic_identity,
                raw_sha256=component.raw_sha256,
                raw_size_bytes=component.raw_size_bytes,
                values=values,
            )
        if not components:
            raise NativeV3V6ProjectionError("v6 multipart resource is empty")
        return NativeV3V6MultipartResource(
            resource_id=resource.resource_id,
            role=resource.resource_role,
            schema=resource.resource_schema,
            semantic_identity=resource.resource_semantic_identity,
            components=MappingProxyType(dict(sorted(components.items()))),
        )


def _json_resource_projection(
    artifact: Any, descriptor: Any, resource_id: str
) -> NativeV3V6JsonResource:
    resource = _descriptor_resource(descriptor, resource_id)
    if resource is None or resource.encoding != "json":
        raise NativeV3V6ProjectionError(f"v6 JSON authority is invalid: {resource_id}")
    try:
        with open_verified_student_resource_v6(
            artifact, resource_id, strict=True
        ) as content:
            payload = json.loads(content.read())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeV3V6ProjectionError(
            f"verified JSON authority cannot be decoded: {resource_id}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise NativeV3V6ProjectionError(
            f"v6 JSON authority must be an object: {resource_id}"
        )
    return NativeV3V6JsonResource(
        resource_id=resource.resource_id,
        role=resource.role,
        schema=resource.schema,
        semantic_identity=resource.semantic_identity,
        raw_sha256=resource.raw_sha256,
        raw_size_bytes=resource.raw_size_bytes,
        content=_freeze(payload),
    )


def _descriptor_resource(descriptor: Any, resource_id: str) -> Any | None:
    resources = (
        *getattr(descriptor, "authority_resources", ()),
        *getattr(descriptor, "non_authority_resources", ()),
    )
    matches = [
        resource for resource in resources if resource.resource_id == resource_id
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _load_component(component: Any, resource_id: str) -> np.ndarray:
    try:
        values = np.array(np.load(component.content, allow_pickle=False), copy=True)
    except (OSError, ValueError) as exc:
        raise NativeV3V6ProjectionError(
            f"verified multipart component cannot be decoded: {resource_id}"
        ) from exc
    if (
        values.dtype.str != component.dtype
        or tuple(values.shape) != tuple(component.shape)
        or values.ndim != len(component.axes)
        or values.dtype.hasobject
    ):
        raise NativeV3V6ProjectionError(
            f"verified multipart component metadata mismatch: {resource_id}"
        )
    values.setflags(write=False)
    return values


def _jsonl_records(artifact: Any, resource_id: str) -> tuple[Mapping[str, Any], ...]:
    with open_verified_student_jsonl_records_v6(
        artifact, resource_id, strict=True
    ) as rows:
        return tuple(_record(row, resource_id) for row in rows)


def _m7_records(artifact: Any, resource_id: str) -> tuple[Mapping[str, Any], ...]:
    with open_verified_student_m7_payload_v6(
        artifact, resource_id, strict=True
    ) as rows:
        records = tuple(_record(row, resource_id) for row in rows)
        if rows.verification_state != "fully_verified":
            raise NativeV3V6ProjectionError("v6 M7 payload was not fully verified")
        return records


def _record(value: Any, resource_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeV3V6ProjectionError(f"v6 record is invalid: {resource_id}")
    return _freeze(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
