"""Anti-cheat coverage for the P3.11.10A case implementation registry."""
# ruff: noqa: E501

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from radjax_student.validation.p3_11_10_gate.gate import execute_case
from radjax_student.validation.p3_11_10_gate.implementations import (
    CASE_IMPLEMENTATIONS,
    GateCaseImplementation,
    validate_implementations,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    observe_failure,
    public_boundary,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES, SECTIONS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.jax
def test_every_inventory_case_has_distinct_case_bound_implementation_and_mutation():
    registry = validate_implementations()
    assert tuple(registry) == tuple(case.case_id for case in CASES)
    assert len({item.identity for item in registry.values()}) == len(CASES)
    assert len({item.behavior_identity for item in registry.values()}) == len(CASES)
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


def test_literal_functions_cannot_receive_inventory_or_expectation_metadata():
    forbidden = ("case_id", "expected_failure", "expected_boundary", "expected_outcome")
    for implementation in CASE_IMPLEMENTATIONS.values():
        function = implementation.function
        assert tuple(inspect.signature(function).parameters) == ("context",)
        source = inspect.getsource(function)
        assert not any(token in source for token in forbidden)


def test_implementation_identity_includes_bound_contract_and_rejects_behavior_reuse(
    monkeypatch: pytest.MonkeyPatch,
):
    first, second = CASES[:2]
    first_implementation = CASE_IMPLEMENTATIONS[first.case_id]
    assert first_implementation.case_id == first.case_id
    assert first_implementation.expected_boundary == first.boundary
    assert first_implementation.execution_class == first.execution_class
    assert first_implementation.preparation_identity
    assert first_implementation.mutation_identity
    assert first_implementation.invocation_identity
    assert first_implementation.observation_adapter_identity
    monkeypatch.setitem(
        CASE_IMPLEMENTATIONS,
        second.case_id,
        first_implementation,
    )
    with pytest.raises(ValueError, match="function_reused|behavior_reused"):
        validate_implementations()


def test_observer_can_only_normalize_the_actual_public_failure():
    @public_boundary("registry_validation")
    def fail(_: object) -> None:
        raise ValueError("unregistered public failure message")

    probe = BoundaryProbe("registry_validation", fail, "0" * 64)
    probe.call_catching(object())
    observed = observe_failure(probe)
    assert observed is not None
    assert observed.code == "unrecognized_public_failure"
    assert tuple(inspect.signature(observe_failure).parameters) == ("probe",)


def test_changing_an_inventory_expectation_cannot_change_an_observed_failure():
    case = next(item for item in CASES if item.expected_outcome == "reject")
    altered = replace(case, expected_failure="foreign_expected_failure")
    result = execute_case(altered, ROOT)
    assert result.observed_failure is not None
    assert result.observed_failure.code != altered.expected_failure
    assert result.classification == "wrong_failure"


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
