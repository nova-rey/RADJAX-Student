from __future__ import annotations

import io
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from radjax_student.artifacts.native_v3_v6 import (
    NativeV3V6ProjectionError,
    open_native_v3_v6_behavioral_projection,
)


def test_v6_projection_requires_strict_before_any_contract_access(monkeypatch) -> None:
    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6.validate_and_resolve_student_consumption",
        lambda *_args, **_kwargs: pytest.fail("must not admit non-strict input"),
    )

    with pytest.raises(NativeV3V6ProjectionError, match="strict=True"):
        open_native_v3_v6_behavioral_projection("artifact", strict=False)


def test_v6_projection_preserves_eos_pad_alias_and_never_retains_locators(
    monkeypatch,
) -> None:
    descriptor = SimpleNamespace(
        language_binding_digest="sha256:" + "a" * 64,
        behavioral_source_identity="sha256:source",
        behavioral_authority_digest="sha256:authority",
        package_semantic_identity="sha256:package",
        composition_digest="sha256:composition",
        authority_resources=_authority_resource_descriptors(),
        non_authority_resources=(),
    )
    language = SimpleNamespace(
        canonical_binding_digest="sha256:" + "a" * 64,
        tokenizer={
            "implementation_id": "test",
            "family": "test",
            "revision": {"value": "sha256:" + "b" * 64},
            "configuration_identity": "sha256:" + "c" * 64,
            "normalization_identity": "sha256:" + "d" * 64,
        },
        vocabulary={
            "vocabulary_size": 2,
            "vocabulary_identity": "sha256:" + "e" * 64,
            "vocabulary_map_digest": "sha256:" + "e" * 64,
            "added_tokens": [],
            "reserved_token_ids": [1],
            "special_tokens": [
                {"name": "eos", "token_id": 1},
                {"name": "pad", "token_id": 1},
            ],
        },
    )
    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6.validate_and_resolve_student_consumption",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, descriptor=descriptor),
    )
    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6.open_verified_student_resource_v6",
        _open_json_resource,
    )
    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6.resolve_student_language_binding",
        lambda *_args, **_kwargs: language,
    )
    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6._multipart_projection",
        lambda _artifact, resource_id: SimpleNamespace(resource_id=resource_id),
    )
    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6._jsonl_records",
        lambda _artifact, resource_id: ({"resource": resource_id},),
    )
    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6._m7_records",
        lambda _artifact, resource_id: ({"resource": resource_id},),
    )

    projection = open_native_v3_v6_behavioral_projection("artifact")

    assert projection.language.special_tokens.eos_token_id == 1
    assert projection.language.special_tokens.pad_token_id == 1
    assert projection.language.schema_version == "hf_language_projection_v1"
    assert projection.authority_reference.content == {
        "resource": "authority_reference/default"
    }
    assert projection.corridor_mode_table.content == {
        "resource": "corridor_mode_table/default"
    }
    assert "locator" not in projection.__dict__
    assert "artifact" not in projection.__dict__


def test_multipart_projection_rejects_component_metadata_mismatch(monkeypatch) -> None:
    component = SimpleNamespace(
        component_id="input_ids",
        dtype="<i4",
        shape=(1,),
        axes=("example",),
        semantic_identity="sha256:component",
        raw_sha256="sha256:raw",
        raw_size_bytes=4,
        content=_npy(np.asarray([1], dtype=np.int64)),
    )
    resource = SimpleNamespace(
        resource_id="target_shard/default",
        resource_role="target_shard",
        resource_schema="schema",
        resource_semantic_identity="sha256:resource",
        components={"input_ids": component},
    )

    @contextmanager
    def open_resource(*_args, **_kwargs):
        yield resource

    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_v6.open_verified_student_multipart_resource_v6",
        open_resource,
    )
    from radjax_student.artifacts.native_v3_v6 import _multipart_projection

    with pytest.raises(NativeV3V6ProjectionError, match="metadata mismatch"):
        _multipart_projection("artifact", "target_shard/default")


def test_v6_authority_closure_rejects_missing_or_substituted_roles() -> None:
    from radjax_student.artifacts.native_v3_v6 import _assert_authority_closure

    resources = _authority_resource_descriptors()
    with pytest.raises(NativeV3V6ProjectionError, match="closure"):
        _assert_authority_closure(SimpleNamespace(authority_resources=resources[:-1]))
    substituted = list(resources)
    substituted[-1] = _resource_descriptor("target_shard/other", "target_shard")
    with pytest.raises(NativeV3V6ProjectionError, match="closure"):
        _assert_authority_closure(
            SimpleNamespace(authority_resources=tuple(substituted))
        )


def _npy(value: np.ndarray):
    content = io.BytesIO()
    np.save(content, value, allow_pickle=False)
    content.seek(0)
    return content


def _authority_resource_descriptors() -> tuple[SimpleNamespace, ...]:
    return tuple(
        _resource_descriptor(resource_id, role)
        for role, resource_id in {
            "authority_reference": "authority_reference/default",
            "corridor_assignment": "corridor_assignment/default",
            "corridor_mode_table": "corridor_mode_table/default",
            "example_registry": "example_registry/default",
            "selected_exemplar_payload": "selected_exemplar_payload/default",
            "selected_passport_index": "selected_passport_index/default",
            "target_shard": "target_shard/default",
        }.items()
    )


def _resource_descriptor(resource_id: str, role: str) -> SimpleNamespace:
    return SimpleNamespace(
        resource_id=resource_id,
        role=role,
        encoding="json",
        schema="radjax_test_json_v6",
        semantic_identity="sha256:semantic",
        raw_sha256="sha256:raw",
        raw_size_bytes=1,
    )


@contextmanager
def _open_json_resource(_artifact, resource_id: str, *, strict: bool):
    assert strict
    yield io.BytesIO(json.dumps({"resource": resource_id}).encode())
