from __future__ import annotations

import io
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from radjax_student.artifacts.native_v3 import NativeV3StudentConsumptionView
from radjax_student.artifacts.native_v3_payloads import (
    load_native_v3_student_payloads,
)


def test_payload_loader_opens_declared_resources_and_freezes_values(
    monkeypatch,
) -> None:
    resources = {
        "target_shard": _resource("target_shard", "npz"),
        "example_registry": _resource("example_registry", "json"),
        "corridor_mode_table": _resource("corridor_mode_table", "json"),
        "corridor_assignment": _resource("corridor_assignment", "npz"),
        "selected_passport_index": _resource("selected_passport_index", "json"),
        "selected_exemplar_payload": _resource("selected_exemplar_payload", "json"),
        "corridor_observed_statistics": _resource(
            "corridor_observed_statistics", "npz"
        ),
        "row_range_declaration": _resource("row_range_declaration", "json"),
        "delivery_receipt": _resource("delivery_receipt", "json"),
        "authority_reference": _resource("authority_reference", "json"),
    }
    contents = {
        "target_shard/default": _npz(
            input_ids=np.asarray([[1, 2]], dtype=np.int32),
            attention_mask=np.asarray([[1, 1]], dtype=np.int32),
            corridor_lengths=np.asarray([2], dtype=np.int32),
        ),
        "example_registry/default": _json(
            {"examples": [{"global_example_index": 0, "selected_example_id": "x"}]}
        ),
        "corridor_mode_table/default": _json({"modes": [{"mode_id": 4}]}),
        "corridor_assignment/default": _npz(
            position_example_index=np.asarray([0], dtype=np.int32),
            position=np.asarray([1], dtype=np.int32),
            mode_id=np.asarray([4], dtype=np.int32),
            weight=np.asarray([0.5], dtype=np.float32),
        ),
        "selected_passport_index/default": _json(
            {
                "selected_exemplars": [
                    {"selected_example_id": "x", "selected_position": 1}
                ]
            }
        ),
        "selected_exemplar_payload/default": _json(
            {"selected_exemplars": [{"selected_example_id": "x", "top_token_ids": [2]}]}
        ),
        "corridor_observed_statistics/default": _npz(
            entropy=np.asarray([1.0], dtype=np.float32),
            top1_margin=np.asarray([0.1], dtype=np.float32),
            top8_mass=np.asarray([0.2], dtype=np.float32),
            top32_mass=np.asarray([0.3], dtype=np.float32),
            tail_mass=np.asarray([0.7], dtype=np.float32),
        ),
        "row_range_declaration/default": _json({"example_count": 1}),
    }
    opened: list[str] = []

    @contextmanager
    def open_resource(_path, resource_id: str, *, strict: bool):
        assert strict
        opened.append(resource_id)
        yield io.BytesIO(contents[resource_id])

    monkeypatch.setattr(
        "radjax_student.artifacts.native_v3_payloads.open_verified_student_resource_v4",
        open_resource,
    )
    descriptor = SimpleNamespace(
        corridor_resources=(
            resources["target_shard"],
            resources["example_registry"],
            resources["corridor_mode_table"],
            resources["corridor_assignment"],
        ),
        exemplar_resources=(
            resources["selected_passport_index"],
            resources["selected_exemplar_payload"],
        ),
        validation_resources=(
            resources["corridor_observed_statistics"],
            resources["row_range_declaration"],
            resources["delivery_receipt"],
            resources["authority_reference"],
        ),
        delivery={"path": "provenance-only"},
        provenance={"source": "contract"},
    )
    consumption = NativeV3StudentConsumptionView(
        artifact_path=SimpleNamespace(),
        contract_assets=SimpleNamespace(),
        descriptor=descriptor,
    )

    payloads = load_native_v3_student_payloads(consumption, strict=True)

    assert opened == [
        "target_shard/default",
        "corridor_assignment/default",
        "corridor_observed_statistics/default",
        "example_registry/default",
        "corridor_mode_table/default",
        "selected_passport_index/default",
        "selected_exemplar_payload/default",
        "row_range_declaration/default",
    ]
    assert payloads.target_shard.input_ids.tolist() == [[1, 2]]
    assert not payloads.target_shard.input_ids.flags.writeable
    assert payloads.corridor_assignments.mode_ids.tolist() == [4]
    assert payloads.selected_passports[0]["selected_position"] == 1
    assert payloads.selected_exemplar_payloads[0]["top_token_ids"] == (2,)
    assert payloads.delivery["path"] == "provenance-only"


def _resource(role: str, encoding: str):
    return SimpleNamespace(
        resource_id=f"{role}/default",
        role=role,
        instance_id="default",
        semantic_digest="sha256:" + "a" * 64,
        raw_sha256="sha256:" + "b" * 64,
        raw_size_bytes=1,
        encoding=encoding,
        classification="batch",
        consumption={"declared": role},
    )


def _json(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def _npz(**arrays: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()
