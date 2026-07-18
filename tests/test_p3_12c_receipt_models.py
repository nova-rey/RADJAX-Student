"""Base-suite strict receipt authority tests for P3.12C."""

from __future__ import annotations

import hashlib

import pytest

from radjax_student.validation.p3_12c_production_lifecycle_assembly import (
    implementation_audit,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly.inventory import (
    ADVERSARIAL_CASE_IDS,
    ADVERSARIAL_CASE_SPECS,
    POSITIVE_CASE_IDS,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly.models import (
    AdversarialResult,
    LifecycleAssemblyProof,
    PositiveResult,
    build_receipt,
    validate_receipt,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _proof() -> LifecycleAssemblyProof:
    audit = implementation_audit.AssemblyAuthorityAudit(
        _sha("source"),
        (
            implementation_audit.AssemblyAuditSourceEntry(
                "src/example.py", _sha("entry")
            ),
        ),
        POSITIVE_CASE_IDS,
        ADVERSARIAL_CASE_IDS,
        (),
    )
    positives = tuple(
        PositiveResult(case_id, "boundary", _sha(case_id))
        for case_id in POSITIVE_CASE_IDS
    )
    adversaries = tuple(
        AdversarialResult(
            case_id=spec.case_id,
            category=spec.category,
            mutation_applied=True,
            baseline_input_digest=_sha(spec.case_id + ".baseline"),
            mutated_input_digest=_sha(spec.case_id + ".mutated"),
            intended_boundary=spec.intended_boundary,
            boundary_callable_identity=spec.intended_boundary,
            observed_boundary=spec.intended_boundary,
            observed_exception_type="ExpectedError",
            expected_code=spec.expected_code,
            observed_code=spec.expected_code,
            deterministic_first_failure=True,
            first_run_evidence_digest=_sha(spec.case_id + ".first"),
            second_run_evidence_digest=_sha(spec.case_id + ".second"),
            outcome="reject",
        )
        for spec in ADVERSARIAL_CASE_SPECS
    )
    return LifecycleAssemblyProof(
        _sha("assembly"),
        positives,
        adversaries,
        audit,
        _sha("checkpoint"),
        _sha("report"),
        _sha("dependency"),
    )


def test_receipt_requires_typed_executed_proof_and_exact_canonical_inventories():
    proof = _proof()
    receipt = build_receipt(proof)
    assert validate_receipt(receipt) == receipt
    assert receipt["positive_proof_count"] == 17
    assert receipt["adversarial_case_count"] == 36
    with pytest.raises(TypeError, match="typed"):
        build_receipt(object())  # type: ignore[arg-type]
    reordered = dict(receipt)
    reordered["positive_case_ids"] = list(reversed(POSITIVE_CASE_IDS))
    with pytest.raises(ValueError, match="positive inventory"):
        validate_receipt(reordered)
    caller_success = dict(receipt) | {"passed": True}
    with pytest.raises(ValueError, match="unknown"):
        validate_receipt(caller_success)


def test_receipt_counts_are_derived_from_adversarial_evidence():
    receipt = build_receipt(_proof())
    mutated = dict(receipt)
    mutated["wrong_failure_count"] = 1
    with pytest.raises(ValueError, match="evidence-derived"):
        validate_receipt(mutated)
