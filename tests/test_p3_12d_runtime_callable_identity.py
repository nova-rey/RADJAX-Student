"""Focused P3.12D executable evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_student.validation.p3_12d_runtime_callable_identity.inventory import (
    ADVERSARIAL_CASE_IDS,
    POSITIVE_CASE_IDS,
)
from radjax_student.validation.p3_12d_runtime_callable_identity.runner_jax import (
    execute_raw_diagnostic,
)


def test_raw_adversaries_are_exact_callable_bound_and_fresh() -> None:
    results = execute_raw_diagnostic()
    assert tuple(item.case_id for item in results) == ADVERSARIAL_CASE_IDS
    assert all(item.mutation_applied for item in results)
    assert all(item.deterministic_first_failure for item in results)
    assert all(
        item.boundary_callable_identity == item.intended_boundary for item in results
    )
    assert all(item.observed_boundary == item.intended_boundary for item in results)
    assert all(item.observed_code == item.expected_code for item in results)
    assert all(item.outcome == "reject" for item in results)


@pytest.mark.jax
def test_product_proof_and_receipt_are_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    from radjax_student.validation.p3_12d_runtime_callable_identity.models import (
        build_receipt,
        validate_receipt,
    )
    from radjax_student.validation.p3_12d_runtime_callable_identity.runner_jax import (
        execute_runtime_callable_identity_proof,
    )

    first = build_receipt(execute_runtime_callable_identity_proof(tmp_path / "first"))
    second = build_receipt(execute_runtime_callable_identity_proof(tmp_path / "second"))
    assert validate_receipt(first) == first
    assert first["positive_proof_count"] == len(POSITIVE_CASE_IDS) == 18
    assert first["adversarial_case_count"] == len(ADVERSARIAL_CASE_IDS) == 40
    assert (
        json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode() == (
        json.dumps(second, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
