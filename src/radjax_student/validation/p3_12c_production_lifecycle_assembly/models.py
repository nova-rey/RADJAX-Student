"""Strict typed executed-evidence models for the P3.12C receipt."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .implementation_audit import (
    SCHEMA_VERSION as AUDIT_SCHEMA_VERSION,
)
from .implementation_audit import (
    AssemblyAuthorityAudit,
)
from .inventory import (
    ADVERSARIAL_CASE_IDS,
    ADVERSARIAL_CASE_SPECS,
    POSITIVE_CASE_IDS,
)

SCHEMA_VERSION = "radjax.p3_12c_production_lifecycle_assembly_receipt.v1"
ASSEMBLY_SCHEMA_VERSION = "radjax.jax_learning_assembly.v1"
REQUEST_SCHEMA_VERSION = "radjax.jax_learning_assembly_request.v1"
RESULT_SCHEMA_VERSION = "radjax.jax_learning_assembly_result.v1"
NON_CLAIMS = (
    "rwkv_not_implemented",
    "phase4_not_started",
    "tome_consumption_not_proven",
    "distillation_not_proven",
    "hf_export_not_implemented",
    "model_quality_not_measured",
    "multi_device_not_proven",
    "tpu_execution_not_proven",
    "performance_not_measured",
    "production_cli_not_implemented",
    "resume_assembly_not_claimed_unless_explicitly_proven",
)
_OUTCOMES = {
    "reject",
    "unexpected_pass",
    "unexpected_failure",
    "wrong_failure",
    "boundary_mismatch",
    "mutation_not_applied",
    "non_deterministic_first_failure",
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        (
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    ).hexdigest()


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class PositiveResult:
    """A successfully executed positive proof; no caller success Boolean exists."""

    case_id: str
    boundary: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("positive case ID is invalid")
        if not isinstance(self.boundary, str) or not self.boundary:
            raise ValueError("positive boundary is invalid")
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
            raise ValueError("positive result fields are invalid")
        return cls(payload["case_id"], payload["boundary"], payload["evidence_digest"])


@dataclass(frozen=True)
class AdversarialResult:
    """Observed evidence from two actual callable-bound fresh invocations."""

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
        for name in (
            "case_id",
            "category",
            "intended_boundary",
            "boundary_callable_identity",
            "observed_boundary",
            "expected_code",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is invalid")
        if self.observed_exception_type is not None and not isinstance(
            self.observed_exception_type, str
        ):
            raise ValueError("observed exception type is invalid")
        if self.observed_code is not None and not isinstance(self.observed_code, str):
            raise ValueError("observed code is invalid")
        if type(self.mutation_applied) is not bool:
            raise ValueError("mutation_applied must be an exact Boolean")
        if type(self.deterministic_first_failure) is not bool:
            raise ValueError("deterministic_first_failure must be an exact Boolean")
        for name in (
            "baseline_input_digest",
            "mutated_input_digest",
            "first_run_evidence_digest",
            "second_run_evidence_digest",
        ):
            _sha(getattr(self, name), name)
        if self.outcome not in _OUTCOMES:
            raise ValueError("adversarial outcome is invalid")
        if self.outcome == "reject" and (
            not self.mutation_applied
            or not self.deterministic_first_failure
            or self.boundary_callable_identity != self.intended_boundary
            or self.observed_boundary != self.intended_boundary
            or self.observed_exception_type is None
            or self.observed_code != self.expected_code
        ):
            raise ValueError("reject outcome must derive from exact observed evidence")

    def to_dict(self) -> dict[str, Any]:
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
        required = {
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
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("adversarial result fields are invalid")
        return cls(**dict(payload))


def error_counts(cases: tuple[AdversarialResult, ...]) -> dict[str, int]:
    """Derive all tolerance counts from observed outcome evidence only."""

    return {
        "mutation_not_applied_count": sum(
            item.outcome == "mutation_not_applied" for item in cases
        ),
        "boundary_mismatch_count": sum(
            item.outcome == "boundary_mismatch" for item in cases
        ),
        "wrong_failure_count": sum(item.outcome == "wrong_failure" for item in cases),
        "non_deterministic_first_failure_count": sum(
            item.outcome == "non_deterministic_first_failure" for item in cases
        ),
        "unexpected_pass_count": sum(
            item.outcome == "unexpected_pass" for item in cases
        ),
        "unexpected_failure_count": sum(
            item.outcome == "unexpected_failure" for item in cases
        ),
    }


@dataclass(frozen=True)
class LifecycleAssemblyProof:
    """One typed executed proof; it cannot carry caller acceptance claims."""

    assembly_digest: str
    positives: tuple[PositiveResult, ...]
    adversaries: tuple[AdversarialResult, ...]
    audit: AssemblyAuthorityAudit
    checkpoint_evidence_digest: str
    report_evidence_digest: str
    dependency_audit_digest: str

    def __post_init__(self) -> None:
        for name in (
            "assembly_digest",
            "checkpoint_evidence_digest",
            "report_evidence_digest",
            "dependency_audit_digest",
        ):
            _sha(getattr(self, name), name)
        positives = tuple(self.positives)
        adversaries = tuple(self.adversaries)
        if not all(isinstance(item, PositiveResult) for item in positives):
            raise TypeError("positive evidence must be PositiveResult")
        if not all(isinstance(item, AdversarialResult) for item in adversaries):
            raise TypeError("adversarial evidence must be AdversarialResult")
        if tuple(item.case_id for item in positives) != POSITIVE_CASE_IDS:
            raise ValueError("P3.12C requires exact ordered positive inventory")
        if tuple(item.case_id for item in adversaries) != ADVERSARIAL_CASE_IDS:
            raise ValueError("P3.12C requires exact ordered adversarial inventory")
        if not isinstance(self.audit, AssemblyAuthorityAudit):
            raise TypeError("P3.12C requires a typed one-authority audit")
        if self.audit.status != "pass":
            raise ValueError("P3.12C requires a passing one-authority audit")
        if (
            self.audit.positive_case_ids != POSITIVE_CASE_IDS
            or self.audit.adversarial_case_ids != ADVERSARIAL_CASE_IDS
            or self.audit.positive_inventory_count != len(POSITIVE_CASE_IDS)
            or self.audit.adversarial_inventory_count != len(ADVERSARIAL_CASE_IDS)
        ):
            raise ValueError("P3.12C proof and source audit inventories disagree")
        object.__setattr__(self, "positives", positives)
        object.__setattr__(self, "adversaries", adversaries)

    @property
    def evidence_digest(self) -> str:
        return digest(
            {
                "assembly_digest": self.assembly_digest,
                "positives": [item.to_dict() for item in self.positives],
                "adversaries": [item.to_dict() for item in self.adversaries],
                "audit": self.audit.to_dict(),
                "checkpoint_evidence_digest": self.checkpoint_evidence_digest,
                "report_evidence_digest": self.report_evidence_digest,
                "dependency_audit_digest": self.dependency_audit_digest,
            }
        )


def build_receipt(proof: LifecycleAssemblyProof) -> dict[str, Any]:
    """Build a pass receipt only from completed typed executed evidence."""

    if not isinstance(proof, LifecycleAssemblyProof):
        raise TypeError("receipt requires typed executed LifecycleAssemblyProof")
    counts = error_counts(proof.adversaries)
    if any(counts.values()) or any(
        item.outcome != "reject" for item in proof.adversaries
    ):
        raise ValueError("cannot build a passing receipt from failed case evidence")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "assembly_schema_version": ASSEMBLY_SCHEMA_VERSION,
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "assembly_digest": proof.assembly_digest,
        "production_assembler_identity": (
            "radjax_student.learning.assemble_jax_learning_lifecycle"
        ),
        "positive_proof_count": len(proof.positives),
        "positive_case_ids": list(POSITIVE_CASE_IDS),
        "positive_results": [item.to_dict() for item in proof.positives],
        "adversarial_case_count": len(proof.adversaries),
        "adversarial_case_ids": list(ADVERSARIAL_CASE_IDS),
        "adversarial_results": [item.to_dict() for item in proof.adversaries],
        **counts,
        "one_authority_audit_schema": AUDIT_SCHEMA_VERSION,
        "one_authority_audit_digest": proof.audit.implementation_audit_digest,
        "dependency_audit_digest": proof.dependency_audit_digest,
        "checkpoint_evidence_digest": proof.checkpoint_evidence_digest,
        "report_evidence_digest": proof.report_evidence_digest,
        "claims_not_made": list(NON_CLAIMS),
    }
    payload["evidence_digest"] = digest(payload)
    return payload


def validate_receipt(payload: object) -> dict[str, Any]:
    """Strictly validate canonical inventories and derived evidence counts."""

    if not isinstance(payload, Mapping):
        raise ValueError("receipt must be an object")
    required = {
        "schema_version",
        "status",
        "assembly_schema_version",
        "request_schema_version",
        "result_schema_version",
        "assembly_digest",
        "production_assembler_identity",
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
        "one_authority_audit_schema",
        "one_authority_audit_digest",
        "dependency_audit_digest",
        "checkpoint_evidence_digest",
        "report_evidence_digest",
        "claims_not_made",
        "evidence_digest",
    }
    if set(payload) != required:
        raise ValueError("receipt fields are missing or unknown")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["assembly_schema_version"] != ASSEMBLY_SCHEMA_VERSION
        or payload["request_schema_version"] != REQUEST_SCHEMA_VERSION
        or payload["result_schema_version"] != RESULT_SCHEMA_VERSION
        or payload["production_assembler_identity"]
        != "radjax_student.learning.assemble_jax_learning_lifecycle"
    ):
        raise ValueError("receipt schema or assembler identity is invalid")
    if tuple(payload["positive_case_ids"]) != POSITIVE_CASE_IDS:
        raise ValueError("receipt positive inventory is invalid")
    if tuple(payload["adversarial_case_ids"]) != ADVERSARIAL_CASE_IDS:
        raise ValueError("receipt adversarial inventory is invalid")
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
        raise ValueError("receipt positive results are invalid")
    if tuple(item.case_id for item in adversaries) != ADVERSARIAL_CASE_IDS:
        raise ValueError("receipt adversarial results are invalid")
    for result, spec in zip(adversaries, ADVERSARIAL_CASE_SPECS, strict=True):
        if (
            result.category != spec.category
            or result.intended_boundary != spec.intended_boundary
            or result.expected_code != spec.expected_code
        ):
            raise ValueError("receipt adversarial authority is invalid")
    if payload["positive_proof_count"] != len(POSITIVE_CASE_IDS) or payload[
        "adversarial_case_count"
    ] != len(ADVERSARIAL_CASE_IDS):
        raise ValueError("receipt inventory count is invalid")
    counts = error_counts(adversaries)
    if any(payload[name] != value for name, value in counts.items()):
        raise ValueError("receipt tolerance counts are not evidence-derived")
    if payload["status"] != "pass" or any(counts.values()):
        raise ValueError("receipt contains failed evidence")
    if any(item.outcome != "reject" for item in adversaries):
        raise ValueError("receipt adversary outcome is not accepting")
    for name in (
        "assembly_digest",
        "one_authority_audit_digest",
        "dependency_audit_digest",
        "checkpoint_evidence_digest",
        "report_evidence_digest",
        "evidence_digest",
    ):
        _sha(payload[name], name)
    if payload["one_authority_audit_schema"] != AUDIT_SCHEMA_VERSION:
        raise ValueError("receipt audit schema is invalid")
    if tuple(payload["claims_not_made"]) != NON_CLAIMS:
        raise ValueError("receipt claims-not-made inventory is invalid")
    without_digest = dict(payload)
    evidence_digest = without_digest.pop("evidence_digest")
    if evidence_digest != digest(without_digest):
        raise ValueError("receipt evidence digest is invalid")
    return dict(payload)


__all__ = [
    "ADVERSARIAL_CASE_IDS",
    "ASSEMBLY_SCHEMA_VERSION",
    "LifecycleAssemblyProof",
    "NON_CLAIMS",
    "POSITIVE_CASE_IDS",
    "PositiveResult",
    "AdversarialResult",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_receipt",
    "digest",
    "error_counts",
    "validate_receipt",
]
