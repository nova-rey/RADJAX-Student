"""Anti-cheat coverage for the P3.11.10A case implementation registry."""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import pytest

from radjax_student.validation.p3_11_10_gate.gate import execute_case
from radjax_student.validation.p3_11_10_gate.implementations import (
    CASE_IMPLEMENTATIONS,
    GateCaseImplementation,
    validate_implementations,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES, SECTIONS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.jax
def test_every_inventory_case_has_distinct_case_bound_implementation_and_mutation():
    registry = validate_implementations()
    assert tuple(registry) == tuple(case.case_id for case in CASES)
    assert len({item.identity for item in registry.values()}) == len(CASES)
    for section in SECTIONS:
        cases = [case for case in CASES if case.section_id == section]
        assert len(cases) == len(
            [
                item
                for item in registry.values()
                if item.case_id in {case.case_id for case in cases}
            ]
        )
        for case in cases:
            result = execute_case(case, ROOT)
            assert result.mutation is not None
            assert result.mutation.descriptor
            assert (
                result.mutation.baseline_digest != result.mutation.mutated_input_digest
            )
            assert result.implementation_identity == registry[case.case_id].identity
            assert result.classification in {"expected_pass", "expected_rejection"}


def test_registry_rejects_missing_and_undeclared_implementations(
    monkeypatch: pytest.MonkeyPatch,
):
    original = dict(CASE_IMPLEMENTATIONS)
    monkeypatch.delitem(CASE_IMPLEMENTATIONS, CASES[0].case_id)
    with pytest.raises(ValueError, match="incomplete"):
        validate_implementations()
    monkeypatch.setattr(
        "radjax_student.validation.p3_11_10_gate.implementations.CASE_IMPLEMENTATIONS",
        {**original, "Z.reject.undeclared": original[CASES[0].case_id]},
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_implementations()


def test_expected_and_observed_failures_are_independent_objects():
    case = next(item for item in CASES if item.expected_outcome == "reject")
    result = execute_case(case, ROOT)
    assert result.observed_failure is not None
    assert result.definition.expected_failure_identity is not result.observed_failure
    assert result.observed_failure.code == result.definition.expected_failure


def test_positive_and_negative_cases_share_the_cli_registry():
    assert all(
        isinstance(item, GateCaseImplementation)
        for item in CASE_IMPLEMENTATIONS.values()
    )
    assert set(CASE_IMPLEMENTATIONS) == {case.case_id for case in CASES}
    assert all(
        CASE_IMPLEMENTATIONS[case.case_id].case_id == case.case_id for case in CASES
    )
