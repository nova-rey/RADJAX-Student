"""Student-side admission and identity checks for the P6.U1 fixture."""

from __future__ import annotations

import json
from pathlib import Path

from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_student.artifacts.native_v3_v6 import (
    open_native_v3_v6_behavioral_projection,
)
from radjax_student.behavior.materialize import materialize_behavioral_batches_v1

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native_v3_student_v6_reduced_burn"
PROFILE = "native_v3_student_v6"


def test_p6_u1_student_fixture_is_strictly_admitted_and_qualified() -> None:
    receipt = json.loads((FIXTURE / "FIXTURE_RECEIPT.json").read_text())
    assert receipt["contract"] == {
        "package": "radjax-contract",
        "release": "0.9.0",
        "commit": "1fa43e1aea2e198511db86dafb0aeefa525d48c7",
    }
    assert receipt["qualification"] == {
        "stable_examples": 64,
        "valid_tokens": 4096,
        "selected_example_ids": 60,
    }
    directory = FIXTURE / "student"
    archive = FIXTURE / "student.tgz"
    directory_result = validate_and_resolve_student_consumption(
        directory, profile_id=PROFILE, strict=True
    )
    archive_result = validate_and_resolve_student_consumption(
        archive, profile_id=PROFILE, strict=True
    )
    assert directory_result.ok and directory_result.descriptor is not None
    assert archive_result.ok and archive_result.descriptor is not None
    for field in (
        "language_binding_digest",
        "behavioral_source_identity",
        "behavioral_authority_digest",
        "package_semantic_identity",
        "composition_digest",
    ):
        assert getattr(directory_result.descriptor, field) == getattr(
            archive_result.descriptor, field
        )

    projection = open_native_v3_v6_behavioral_projection(directory)
    materialization = materialize_behavioral_batches_v1(projection)
    assert len(projection.example_registry) == 64
    assert len(projection.selected_passports) == 64
    assert len(projection.selected_exemplar_payloads) == 64
    assert (
        len({str(row["selected_example_id"]) for row in projection.selected_passports})
        == 60
    )
    assert (
        int(projection.target_shard.components["attention_mask"].values.sum()) == 4096
    )
    assert len(materialization.split.assignments) == 64

    stable_ids = {str(row["example_id"]) for row in projection.example_registry}
    passport_keys = {
        (str(row["selected_example_id"]), int(row["selected_position"]))
        for row in projection.selected_passports
    }
    payload_keys = {
        (str(row["selected_example_id"]), int(row["selected_position"]))
        for row in projection.selected_exemplar_payloads
    }
    assert payload_keys == passport_keys
    assert {key[0] for key in passport_keys} <= stable_ids
    for row in projection.selected_exemplar_payloads:
        assert row["payload_ref"]["c5_authoritative_coordinate"] is True
        assert row["payload_ref"]["source_position"] == row["selected_position"]
        assert row["source_score"] == row["selected_score"]
        assert isinstance(row["score_top_token_id"], int)
        assert row["top_token_ids"]


def test_p6_u1_student_profile_has_no_host_or_private_leakage() -> None:
    for path in (FIXTURE / "student").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}:
            text = path.read_text(encoding="utf-8")
            assert "/Users/" not in text
            assert "/var/folders/" not in text
            assert "/tmp/" not in text
            assert ".staging" not in text
        if path.is_file():
            assert (
                not path.relative_to(FIXTURE / "student")
                .as_posix()
                .startswith(("c6/", "reports/", ".staging"))
            )
