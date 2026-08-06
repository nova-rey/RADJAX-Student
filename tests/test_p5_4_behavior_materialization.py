from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from radjax_student.artifacts.native_v3_v6 import (
    NativeV3V6BehavioralProjection,
    NativeV3V6Component,
    NativeV3V6JsonResource,
    NativeV3V6MultipartResource,
)
from radjax_student.behavior import (
    BehavioralMaterializationError,
    BehaviorSplitError,
    BehaviorSplitPolicyV1,
    materialize_behavioral_batches_v1,
)
from radjax_student.hf.language_projection import HFLanguageProjectionV1


def test_p5_4_fixture_split_is_literal_and_leakage_free() -> None:
    batches = materialize_behavioral_batches_v1(_projection())

    assert batches.split.training_example_ids == (
        "corpus_000000001",
        "corpus_000000002",
    )
    assert batches.split.held_out_example_ids == (
        "corpus_000000003",
        "corpus_000000004",
    )
    assert set(batches.training_corridor.example_ids).isdisjoint(
        batches.held_out_corridor.example_ids
    )
    assert batches.training_exemplars.example_ids == ("corpus_000000001",)
    assert batches.held_out_exemplars.example_ids == (
        "corpus_000000003",
        "corpus_000000003",
    )
    assert batches.training_corridor.input_ids.flags.writeable is False
    assert (
        batches.training_exemplars.sparse_targets[0].token_ids.flags.writeable is False
    )
    bounds = batches.training_corridor.mode_bounds
    assert bounds.declared_mode_ids == (0,)
    assert set(bounds.statistics_by_mode[0]) == {
        "entropy",
        "tail_mass",
        "top1_margin",
        "top8_mass",
        "top32_mass",
    }
    assert bounds.statistics_by_mode[0]["entropy"].minimum == 0.0
    assert bounds.statistics_by_mode[0]["entropy"].mean == 0.0
    assert bounds.statistics_by_mode[0]["entropy"].maximum == 0.0


def test_split_policy_is_utf8_stable_identity_bound_and_fails_closed() -> None:
    policy = BehaviorSplitPolicyV1()
    result = policy.split(
        behavioral_source_identity="sha256:" + "1" * 64,
        example_ids=("z", "a", "é", "b"),
        exemplar_example_ids=("é", "a", "é"),
    )

    assert tuple(result.assignments) == ("a", "b", "z", "é")
    assert result.assignments["a"] == "training"
    assert result.assignments["é"] == "held_out"
    assert result.split_identity.startswith("sha256:")
    with pytest.raises(BehaviorSplitError, match="at least two"):
        policy.split(
            behavioral_source_identity="sha256:" + "1" * 64,
            example_ids=("a",),
            exemplar_example_ids=("a",),
        )
    with pytest.raises(BehaviorSplitError, match="duplicate"):
        policy.split(
            behavioral_source_identity="sha256:" + "1" * 64,
            example_ids=("a", "a"),
            exemplar_example_ids=("a", "b"),
        )
    for malformed in ("sha256:" + "g" * 64, "sha256:" + "a" * 63, "a" * 64):
        with pytest.raises(BehaviorSplitError, match="sha256"):
            policy.split(
                behavioral_source_identity=malformed,
                example_ids=("a", "b"),
                exemplar_example_ids=("a", "b"),
            )


def test_materializer_rejects_unjoined_exemplars_and_invalid_coordinates() -> None:
    projection = _projection()
    broken = _replace_projection(
        projection,
        selected_exemplar_payloads=(projection.selected_exemplar_payloads[0],),
    )
    with pytest.raises(BehavioralMaterializationError, match="incomplete"):
        materialize_behavioral_batches_v1(broken)

    components = dict(projection.corridor_assignment.components)
    components["position"] = _component("position", np.asarray([99, 1, 2, 3]))
    broken_assignment = NativeV3V6MultipartResource(
        resource_id="corridor_assignment/default",
        role="corridor_assignment",
        schema="test",
        semantic_identity="sha256:assignment",
        components=MappingProxyType(components),
    )
    with pytest.raises(BehavioralMaterializationError, match="coordinates"):
        materialize_behavioral_batches_v1(
            _replace_projection(projection, corridor_assignment=broken_assignment)
        )

    incomplete_modes = NativeV3V6JsonResource(
        resource_id="corridor_mode_table/default",
        role="corridor_mode_table",
        schema="test",
        semantic_identity="sha256:modes",
        raw_sha256="sha256:raw",
        raw_size_bytes=1,
        content=MappingProxyType(
            {"modes": (MappingProxyType({"mode_id": 0, "statistics": {}}),)}
        ),
    )
    with pytest.raises(BehavioralMaterializationError, match="mode statistics"):
        materialize_behavioral_batches_v1(
            _replace_projection(projection, corridor_mode_table=incomplete_modes)
        )


def test_real_tome_directory_and_archive_are_completely_leakage_free() -> None:
    """Exercise the committed ordinary fixture through public P5.3 projection."""

    fixture = _real_fixture_root()
    if fixture is None:
        pytest.skip("RADJAX Tome v6 fixture is not available in this checkout")
    from radjax_student.artifacts.native_v3_v6 import (
        open_native_v3_v6_behavioral_projection,
    )

    results = tuple(
        materialize_behavioral_batches_v1(
            open_native_v3_v6_behavioral_projection(fixture / delivery)
        )
        for delivery in ("student", "student.tgz")
    )
    directory, archive = results
    for batches in results:
        assert batches.split.training_example_ids == (
            "corpus_000000001",
            "corpus_000000002",
        )
        assert batches.split.held_out_example_ids == (
            "corpus_000000003",
            "corpus_000000004",
        )
        _assert_partition_coverage(batches, "training")
        _assert_partition_coverage(batches, "held_out")
    assert directory.split == archive.split
    assert _batch_coordinates(directory.training_corridor) == _batch_coordinates(
        archive.training_corridor
    )
    assert _batch_coordinates(directory.held_out_corridor) == _batch_coordinates(
        archive.held_out_corridor
    )
    assert _exemplar_keys(directory.training_exemplars) == _exemplar_keys(
        archive.training_exemplars
    )
    assert _exemplar_keys(directory.held_out_exemplars) == _exemplar_keys(
        archive.held_out_exemplars
    )


def _real_fixture_root() -> Path | None:
    configured = os.environ.get("RADJAX_TOME_V6_FIXTURE")
    candidates = (
        Path(configured) if configured else None,
        Path(
            "/Users/Cooper/code/RADJAX-Tome/tests/fixtures/native_v3_student_v6_smoke"
        ),
    )
    return next(
        (
            candidate
            for candidate in candidates
            if candidate is not None
            and (candidate / "student").is_dir()
            and (candidate / "student.tgz").is_file()
        ),
        None,
    )


def _assert_partition_coverage(batches, partition: str) -> None:
    corridor = getattr(batches, f"{partition}_corridor")
    exemplars = getattr(batches, f"{partition}_exemplars")
    allowed = set(getattr(batches.split, f"{partition}_example_ids"))
    assert set(corridor.example_ids) == allowed
    expected_coordinates = {
        "corpus_000000001": 0,
        "corpus_000000002": 1,
        "corpus_000000003": 2,
        "corpus_000000004": 3,
    }
    assert set(corridor.example_indices.tolist()) <= {
        expected_coordinates[example_id] for example_id in allowed
    }
    assert set(exemplars.example_ids) <= allowed
    assert {
        passport["selected_example_id"] for passport in exemplars.passports
    } <= allowed
    assert {passport["selected_example_id"] for passport in exemplars.passports} == set(
        exemplars.example_ids
    )


def _batch_coordinates(batch) -> tuple[tuple[int, int, int, float], ...]:
    return tuple(
        zip(
            batch.example_indices.tolist(),
            batch.positions.tolist(),
            batch.mode_ids.tolist(),
            batch.assignment_weights.tolist(),
            strict=True,
        )
    )


def _exemplar_keys(batch) -> tuple[tuple[object, object, object], ...]:
    return tuple(
        (
            passport["selected_example_id"],
            passport["selected_position"],
            passport["corridor_fingerprint_id"],
        )
        for passport in batch.passports
    )


def _projection() -> NativeV3V6BehavioralProjection:
    target = NativeV3V6MultipartResource(
        resource_id="target_shard/default",
        role="target_shard",
        schema="test",
        semantic_identity="sha256:target",
        components=MappingProxyType(
            {
                "input_ids": _component(
                    "input_ids", np.arange(16, dtype=np.int32).reshape(4, 4)
                ),
                "attention_mask": _component(
                    "attention_mask", np.ones((4, 4), dtype=np.int8)
                ),
            }
        ),
    )
    assignments = NativeV3V6MultipartResource(
        resource_id="corridor_assignment/default",
        role="corridor_assignment",
        schema="test",
        semantic_identity="sha256:assignment",
        components=MappingProxyType(
            {
                "example_index": _component(
                    "example_index", np.asarray([0, 1, 2, 3], dtype=np.int32)
                ),
                "position": _component(
                    "position", np.asarray([0, 1, 2, 3], dtype=np.int32)
                ),
                "mode_id": _component(
                    "mode_id", np.asarray([0, 0, 0, 0], dtype=np.int32)
                ),
                "weight": _component(
                    "weight", np.asarray([1, 1, 1, 1], dtype=np.float32)
                ),
            }
        ),
    )
    passports = tuple(
        MappingProxyType(
            {
                "selected_example_id": example_id,
                "selected_position": position,
                "corridor_fingerprint_id": fingerprint,
            }
        )
        for example_id, position, fingerprint in (
            ("corpus_000000003", 0, "c"),
            ("corpus_000000003", 3, "b"),
            ("corpus_000000001", 2, "a"),
        )
    )
    payloads = tuple(
        MappingProxyType(
            {
                **passport,
                "top_token_ids": (1, 2),
                "top_probs": (0.4, 0.3),
                "tail_mass": 0.3,
            }
        )
        for passport in passports
    )
    modes = NativeV3V6JsonResource(
        resource_id="corridor_mode_table/default",
        role="corridor_mode_table",
        schema="test",
        semantic_identity="sha256:modes",
        raw_sha256="sha256:raw",
        raw_size_bytes=1,
        content=MappingProxyType({"modes": (_mode_declaration(0),)}),
    )
    return NativeV3V6BehavioralProjection(
        language=_language(),
        behavioral_source_identity="sha256:" + "a" * 64,
        behavioral_authority_digest="sha256:authority",
        package_semantic_identity="sha256:package",
        composition_digest="sha256:composition",
        authority_reference=modes,
        corridor_mode_table=modes,
        target_shard=target,
        corridor_assignment=assignments,
        example_registry=tuple(
            MappingProxyType({"example_id": f"corpus_{index:09d}"})
            for index in range(1, 5)
        ),
        selected_passports=passports,
        selected_exemplar_payloads=payloads,
    )


def _component(component_id: str, values: np.ndarray) -> NativeV3V6Component:
    copy = np.array(values, copy=True)
    copy.setflags(write=False)
    return NativeV3V6Component(
        component_id=component_id,
        dtype=copy.dtype.str,
        shape=copy.shape,
        axes=tuple(f"axis_{index}" for index in range(copy.ndim)),
        semantic_identity="sha256:component",
        raw_sha256="sha256:raw",
        raw_size_bytes=copy.nbytes,
        values=copy,
    )


def _mode_declaration(mode_id: int) -> MappingProxyType[str, object]:
    statistics = {
        name: MappingProxyType({"min": 0.0, "mean": 0.0, "max": 0.0})
        for name in ("entropy", "tail_mass", "top1_margin", "top8_mass", "top32_mass")
    }
    return MappingProxyType(
        {"mode_id": mode_id, "statistics": MappingProxyType(statistics)}
    )


def _language() -> HFLanguageProjectionV1:
    from radjax_student.contracts.hf import (
        HFSpecialTokenIdentity,
        HFTokenizerIdentity,
        HFVocabularyIdentity,
    )

    return HFLanguageProjectionV1(
        schema_version="hf_language_projection_v1",
        canonical_binding_digest="sha256:" + "b" * 64,
        tokenizer=HFTokenizerIdentity(
            "test", "test", "a" * 64, "b" * 64, "test", "c" * 64
        ),
        vocabulary=HFVocabularyIdentity(4, "a" * 64, "b" * 64, "c" * 64, None),
        special_tokens=HFSpecialTokenIdentity(None, 1, 1, None, None),
    )


def _replace_projection(
    projection: NativeV3V6BehavioralProjection, **changes: object
) -> NativeV3V6BehavioralProjection:
    values = projection.__dict__ | changes
    return NativeV3V6BehavioralProjection(**values)
