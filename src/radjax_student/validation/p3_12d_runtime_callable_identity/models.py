"""Strict typed P3.12D execution evidence models.

Receipt writing intentionally lives elsewhere: these models cannot manufacture
passing evidence from caller flags.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .inventory import ADVERSARIAL_CASE_IDS, POSITIVE_CASE_IDS

PROOF_SCHEMA_VERSION = "radjax.p3_12d_runtime_callable_identity_proof.v1"
RECEIPT_SCHEMA_VERSION = "radjax.p3_12d_runtime_callable_identity_receipt.v1"
_OUTCOMES = {
    "reject",
    "unexpected_pass",
    "unexpected_failure",
    "wrong_failure",
    "boundary_mismatch",
    "mutation_not_applied",
    "non_deterministic_first_failure",
}
CLAIMS_NOT_MADE = (
    "arbitrary_python_callable_identity_not_supported",
    "arbitrary_closure_identity_not_supported",
    "arbitrary_decorator_identity_not_supported",
    "transitive_dependency_semantics_not_fully_hashed",
    "compiled_executable_serialization_not_implemented",
    "persistent_compilation_cache_not_implemented",
    "cross_process_cache_reuse_not_proven",
    "cross_machine_executable_portability_not_proven",
    "multi_device_execution_not_proven",
    "distributed_execution_not_proven",
    "tpu_execution_not_proven",
    "performance_not_measured",
    "rwkv_not_implemented",
    "phase4_not_started",
    "model_quality_not_measured",
    "hf_export_not_implemented",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class PositiveResult:
    case_id: str
    boundary: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.case_id not in POSITIVE_CASE_IDS or not self.boundary:
            raise ValueError("positive result is invalid")
        _sha(self.evidence_digest, "positive evidence digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "boundary": self.boundary,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, payload: object) -> PositiveResult:
        if not isinstance(payload, Mapping) or set(payload) != {
            "case_id",
            "boundary",
            "evidence_digest",
        }:
            raise ValueError("positive result fields are missing or unknown")
        return cls(payload["case_id"], payload["boundary"], payload["evidence_digest"])


@dataclass(frozen=True)
class AdversarialResult:
    case_id: str
    category: str
    mutation_applied: bool
    baseline_input_digest: str
    mutated_input_digest: str
    intended_boundary: str
    boundary_callable_identity: str
    observed_boundary: str
    observed_exception_type: str | None
    expected_code: str
    observed_code: str | None
    deterministic_first_failure: bool
    first_run_evidence_digest: str
    second_run_evidence_digest: str
    outcome: Literal[
        "reject",
        "unexpected_pass",
        "unexpected_failure",
        "wrong_failure",
        "boundary_mismatch",
        "mutation_not_applied",
        "non_deterministic_first_failure",
    ]

    def __post_init__(self) -> None:
        if self.case_id not in ADVERSARIAL_CASE_IDS:
            raise ValueError("adversarial case ID is invalid")
        if not isinstance(self.mutation_applied, bool):
            raise TypeError("mutation_applied must be boolean")
        if not all(
            isinstance(getattr(self, name), str) and getattr(self, name)
            for name in (
                "category",
                "intended_boundary",
                "boundary_callable_identity",
                "observed_boundary",
                "expected_code",
            )
        ):
            raise ValueError("adversarial boundary and expected code are required")
        if self.observed_code is not None and not self.observed_code:
            raise ValueError("observed code must be nonempty when present")
        if not isinstance(self.deterministic_first_failure, bool):
            raise TypeError("deterministic_first_failure must be boolean")
        if self.observed_exception_type is not None and not isinstance(
            self.observed_exception_type, str
        ):
            raise ValueError("observed exception type must be a string or None")
        if self.outcome not in _OUTCOMES:
            raise ValueError("adversarial outcome is invalid")
        for name in (
            "baseline_input_digest",
            "mutated_input_digest",
            "first_run_evidence_digest",
            "second_run_evidence_digest",
        ):
            _sha(getattr(self, name), name)
        if self.outcome == "reject" and (
            not self.mutation_applied
            or not self.deterministic_first_failure
            or self.boundary_callable_identity != self.intended_boundary
            or self.observed_boundary != self.intended_boundary
            or self.observed_exception_type is None
            or self.observed_code != self.expected_code
        ):
            raise ValueError("reject outcome must derive from exact observed evidence")

    @property
    def exact(self) -> bool:
        return self.observed_code == self.expected_code

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "mutation_applied": self.mutation_applied,
            "baseline_input_digest": self.baseline_input_digest,
            "mutated_input_digest": self.mutated_input_digest,
            "intended_boundary": self.intended_boundary,
            "boundary_callable_identity": self.boundary_callable_identity,
            "observed_boundary": self.observed_boundary,
            "observed_exception_type": self.observed_exception_type,
            "expected_code": self.expected_code,
            "observed_code": self.observed_code,
            "deterministic_first_failure": self.deterministic_first_failure,
            "first_run_evidence_digest": self.first_run_evidence_digest,
            "second_run_evidence_digest": self.second_run_evidence_digest,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, payload: object) -> AdversarialResult:
        fields = {
            "case_id",
            "category",
            "mutation_applied",
            "baseline_input_digest",
            "mutated_input_digest",
            "intended_boundary",
            "boundary_callable_identity",
            "observed_boundary",
            "observed_exception_type",
            "expected_code",
            "observed_code",
            "deterministic_first_failure",
            "first_run_evidence_digest",
            "second_run_evidence_digest",
            "outcome",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("adversarial result fields are missing or unknown")
        return cls(**dict(payload))


@dataclass(frozen=True)
class RuntimeCallableIdentityProof:
    callable_identity_digest: str
    eager_prepared_execution_digest: str
    jit_prepared_execution_digest: str
    positive_results: tuple[PositiveResult, ...]
    adversarial_results: tuple[AdversarialResult, ...]
    callable_identity_audit_digest: str
    checkpoint_evidence_digest: str
    report_evidence_digest: str
    dependency_audit_digest: str
    initialization_rng_identity_digest: str

    def __post_init__(self) -> None:
        for name in (
            "callable_identity_digest",
            "eager_prepared_execution_digest",
            "jit_prepared_execution_digest",
            "callable_identity_audit_digest",
            "checkpoint_evidence_digest",
            "report_evidence_digest",
            "dependency_audit_digest",
            "initialization_rng_identity_digest",
        ):
            _sha(getattr(self, name), name)
        positives = tuple(self.positive_results)
        adversarial = tuple(self.adversarial_results)
        if tuple(item.case_id for item in positives) != POSITIVE_CASE_IDS:
            raise ValueError("positive proof inventory is not exact and ordered")
        if tuple(item.case_id for item in adversarial) != ADVERSARIAL_CASE_IDS:
            raise ValueError("adversarial inventory is not exact and ordered")
        object.__setattr__(self, "positive_results", positives)
        object.__setattr__(self, "adversarial_results", adversarial)

    @property
    def mutation_not_applied_count(self) -> int:
        return sum(
            item.outcome == "mutation_not_applied" for item in self.adversarial_results
        )

    @property
    def boundary_mismatch_count(self) -> int:
        return sum(
            item.outcome == "boundary_mismatch" for item in self.adversarial_results
        )

    @property
    def wrong_failure_count(self) -> int:
        return sum(item.outcome == "wrong_failure" for item in self.adversarial_results)

    @property
    def non_deterministic_first_failure_count(self) -> int:
        return sum(
            item.outcome == "non_deterministic_first_failure"
            for item in self.adversarial_results
        )

    @property
    def unexpected_pass_count(self) -> int:
        return sum(
            item.outcome == "unexpected_pass" for item in self.adversarial_results
        )

    @property
    def unexpected_failure_count(self) -> int:
        return sum(
            item.outcome == "unexpected_failure" for item in self.adversarial_results
        )

    @property
    def status(self) -> str:
        counts = (
            self.mutation_not_applied_count,
            self.boundary_mismatch_count,
            self.wrong_failure_count,
            self.non_deterministic_first_failure_count,
            self.unexpected_pass_count,
            self.unexpected_failure_count,
        )
        return "pass" if not any(counts) else "fail"

    @property
    def evidence_digest(self) -> str:
        return _digest(
            {
                "schema_version": PROOF_SCHEMA_VERSION,
                "callable_identity_digest": self.callable_identity_digest,
                "eager_prepared_execution_digest": self.eager_prepared_execution_digest,
                "jit_prepared_execution_digest": self.jit_prepared_execution_digest,
                "positive_results": [item.to_dict() for item in self.positive_results],
                "adversarial_results": [
                    item.to_dict() for item in self.adversarial_results
                ],
                "callable_identity_audit_digest": self.callable_identity_audit_digest,
                "checkpoint_evidence_digest": self.checkpoint_evidence_digest,
                "report_evidence_digest": self.report_evidence_digest,
                "dependency_audit_digest": self.dependency_audit_digest,
                "initialization_rng_identity_digest": (
                    self.initialization_rng_identity_digest
                ),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PROOF_SCHEMA_VERSION,
            "status": self.status,
            "callable_identity_digest": self.callable_identity_digest,
            "eager_prepared_execution_digest": self.eager_prepared_execution_digest,
            "jit_prepared_execution_digest": self.jit_prepared_execution_digest,
            "positive_proof_count": len(self.positive_results),
            "positive_case_ids": [item.case_id for item in self.positive_results],
            "positive_results": [item.to_dict() for item in self.positive_results],
            "adversarial_case_count": len(self.adversarial_results),
            "adversarial_case_ids": [item.case_id for item in self.adversarial_results],
            "adversarial_results": [
                item.to_dict() for item in self.adversarial_results
            ],
            "mutation_not_applied_count": self.mutation_not_applied_count,
            "boundary_mismatch_count": self.boundary_mismatch_count,
            "wrong_failure_count": self.wrong_failure_count,
            "non_deterministic_first_failure_count": (
                self.non_deterministic_first_failure_count
            ),
            "unexpected_pass_count": self.unexpected_pass_count,
            "unexpected_failure_count": self.unexpected_failure_count,
            "callable_identity_audit_digest": self.callable_identity_audit_digest,
            "checkpoint_evidence_digest": self.checkpoint_evidence_digest,
            "report_evidence_digest": self.report_evidence_digest,
            "dependency_audit_digest": self.dependency_audit_digest,
            "initialization_rng_identity_digest": (
                self.initialization_rng_identity_digest
            ),
            "evidence_digest": self.evidence_digest,
        }


def build_receipt(proof: RuntimeCallableIdentityProof) -> dict[str, object]:
    """Build a receipt only from a typed, accepting executed proof."""
    if not isinstance(proof, RuntimeCallableIdentityProof):
        raise TypeError("receipt requires RuntimeCallableIdentityProof")
    if proof.status != "pass":
        raise ValueError("receipt cannot be built from failed executed evidence")
    from radjax_student.runtime.callables import (
        CALLABLE_DECLARATION_SCHEMA_VERSION,
        CALLABLE_IDENTITY_SCHEMA_VERSION,
        CALLABLE_REFERENCE_SCHEMA_VERSION,
        PREPARED_IDENTITY_SCHEMA_VERSION,
    )

    from .implementation_audit import SCHEMA_VERSION as AUDIT_SCHEMA_VERSION

    receipt = proof.to_dict() | {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "callable_declaration_schema_version": CALLABLE_DECLARATION_SCHEMA_VERSION,
        "callable_identity_schema_version": CALLABLE_IDENTITY_SCHEMA_VERSION,
        "callable_reference_schema_version": CALLABLE_REFERENCE_SCHEMA_VERSION,
        "prepared_execution_identity_schema_version": PREPARED_IDENTITY_SCHEMA_VERSION,
        "callable_identity_audit_schema": AUDIT_SCHEMA_VERSION,
        "claims_not_made": list(CLAIMS_NOT_MADE),
    }
    receipt["evidence_digest"] = _digest(
        {key: value for key, value in receipt.items() if key != "evidence_digest"}
    )
    return receipt


def validate_receipt(payload: object) -> dict[str, object]:
    """Strictly validate the full typed receipt shape and evidence-derived totals."""
    if not isinstance(payload, dict):
        raise ValueError("receipt must be an object")
    required = {
        "schema_version",
        "status",
        "callable_declaration_schema_version",
        "callable_identity_schema_version",
        "callable_reference_schema_version",
        "prepared_execution_identity_schema_version",
        "callable_identity_digest",
        "eager_prepared_execution_digest",
        "jit_prepared_execution_digest",
        "positive_proof_count",
        "positive_case_ids",
        "positive_results",
        "adversarial_case_count",
        "adversarial_case_ids",
        "adversarial_results",
        "mutation_not_applied_count",
        "boundary_mismatch_count",
        "wrong_failure_count",
        "non_deterministic_first_failure_count",
        "unexpected_pass_count",
        "unexpected_failure_count",
        "callable_identity_audit_schema",
        "callable_identity_audit_digest",
        "dependency_audit_digest",
        "checkpoint_evidence_digest",
        "report_evidence_digest",
        "initialization_rng_identity_digest",
        "claims_not_made",
        "evidence_digest",
    }
    if set(payload) != required:
        raise ValueError("receipt fields are missing or unknown")
    if payload["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("receipt schema is invalid")
    if payload["status"] != "pass":
        raise ValueError("receipt status must derive from accepting evidence")
    if tuple(payload["positive_case_ids"]) != POSITIVE_CASE_IDS:
        raise ValueError("receipt positive inventory is not exact and ordered")
    if tuple(payload["adversarial_case_ids"]) != ADVERSARIAL_CASE_IDS:
        raise ValueError("receipt adversarial inventory is not exact and ordered")
    if payload["positive_proof_count"] != 18 or payload["adversarial_case_count"] != 40:
        raise ValueError("receipt inventory counts are invalid")
    if tuple(payload["claims_not_made"]) != CLAIMS_NOT_MADE:
        raise ValueError("receipt claims-not-made inventory is invalid")
    for name in (
        "callable_identity_digest",
        "eager_prepared_execution_digest",
        "jit_prepared_execution_digest",
        "callable_identity_audit_digest",
        "dependency_audit_digest",
        "checkpoint_evidence_digest",
        "report_evidence_digest",
        "initialization_rng_identity_digest",
        "evidence_digest",
    ):
        _sha(payload[name], name)
    try:
        positives = tuple(
            PositiveResult.from_dict(item) for item in payload["positive_results"]
        )
        adversaries = tuple(
            AdversarialResult.from_dict(item) for item in payload["adversarial_results"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("receipt result evidence is malformed") from exc
    if tuple(item.case_id for item in positives) != POSITIVE_CASE_IDS:
        raise ValueError("receipt positive results are not exact and ordered")
    if tuple(item.case_id for item in adversaries) != ADVERSARIAL_CASE_IDS:
        raise ValueError("receipt adversarial results are not exact and ordered")
    counts = {
        "mutation_not_applied_count": sum(
            item.outcome == "mutation_not_applied" for item in adversaries
        ),
        "boundary_mismatch_count": sum(
            item.outcome == "boundary_mismatch" for item in adversaries
        ),
        "wrong_failure_count": sum(
            item.outcome == "wrong_failure" for item in adversaries
        ),
        "non_deterministic_first_failure_count": sum(
            item.outcome == "non_deterministic_first_failure" for item in adversaries
        ),
        "unexpected_pass_count": sum(
            item.outcome == "unexpected_pass" for item in adversaries
        ),
        "unexpected_failure_count": sum(
            item.outcome == "unexpected_failure" for item in adversaries
        ),
    }
    if any(payload[name] != count for name, count in counts.items()):
        raise ValueError("receipt tolerance counts are not evidence-derived")
    if any(item.outcome != "reject" for item in adversaries):
        raise ValueError("receipt adversarial evidence is not accepting")
    if any(
        payload[name] != 0
        for name in (
            "mutation_not_applied_count",
            "boundary_mismatch_count",
            "wrong_failure_count",
            "non_deterministic_first_failure_count",
            "unexpected_pass_count",
            "unexpected_failure_count",
        )
    ):
        raise ValueError("receipt tolerance counts are not accepting")
    calculated = _digest(
        {key: value for key, value in payload.items() if key != "evidence_digest"}
    )
    if payload["evidence_digest"] != calculated:
        raise ValueError("receipt evidence digest is invalid")
    return payload


@dataclass(frozen=True)
class RuntimeCallableIdentityReceipt:
    """Strict serialized receipt wrapper; construction validates every field."""

    payload: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(validate_receipt(self.payload)))

    @classmethod
    def from_proof(
        cls, proof: RuntimeCallableIdentityProof
    ) -> RuntimeCallableIdentityReceipt:
        return cls(build_receipt(proof))

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)
