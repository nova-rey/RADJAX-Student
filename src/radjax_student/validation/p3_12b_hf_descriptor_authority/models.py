"""Strict, JAX-free evidence contracts for the P3.12B.1 acceptance gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from radjax_student.contracts import HFCompatibilityDescriptor

from .implementation_audit import HFDescriptorGateImplementationAudit

SCHEMA_VERSION = "radjax.p3_12b_hf_descriptor_authority.v2"
ADVERSARIAL_CASE_COUNT = 77
POSITIVE_PROOF_COUNT = 22
POSITIVE_CASE_IDS = (
    "descriptor_constructed",
    "reference_derived",
    "canonical_round_trip",
    "construction_determinism",
    "projection_covers_layout",
    "projection_matches_materialized",
    "exportable_keys",
    "tokenizer_complete",
    "vocabulary_complete",
    "special_tokens_valid",
    "lifecycle_binds_descriptor",
    "checkpoint_persists_descriptor",
    "caller_bound_restore",
    "historical_non_resumable",
    "eager_resume_identity",
    "jit_resume_identity",
    "replay_ab_identity",
    "report_summary",
    "report_is_compact",
    "no_export_claim",
    "one_authority_audit",
    "recorded_determinism",
)


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
class HFPositiveProof:
    case_id: str
    boundary: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("positive case_id must be nonempty")
        if not isinstance(self.boundary, str) or not self.boundary:
            raise ValueError("positive boundary must be nonempty")
        _sha(self.evidence_digest, "positive evidence_digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "boundary": self.boundary,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class HFAdversarialResult:
    """Evidence collected after, never before, a literal boundary invocation."""

    case_id: str
    category: str
    intended_boundary: str
    observed_boundary: str
    boundary_callable_identity: str
    baseline_input_digest: str
    mutated_input_digest: str
    mutation_applied: bool
    expected_code: str | None
    observed_code: str | None
    observed_exception_type: str | None
    observed_details_digest: str
    first_run_evidence_digest: str
    second_run_evidence_digest: str
    deterministic_first_failure: bool
    outcome: Literal[
        "reject",
        "invariant_preserved",
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
            "observed_boundary",
            "boundary_callable_identity",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        for name in (
            "baseline_input_digest",
            "mutated_input_digest",
            "observed_details_digest",
            "first_run_evidence_digest",
            "second_run_evidence_digest",
        ):
            _sha(getattr(self, name), name)
        if (
            type(self.mutation_applied) is not bool
            or type(self.deterministic_first_failure) is not bool
        ):
            raise ValueError("adversarial booleans must be exact booleans")
        if self.outcome not in {
            "reject",
            "invariant_preserved",
            "unexpected_pass",
            "unexpected_failure",
            "wrong_failure",
            "boundary_mismatch",
            "mutation_not_applied",
            "non_deterministic_first_failure",
        }:
            raise ValueError("adversarial outcome is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "intended_boundary": self.intended_boundary,
            "observed_boundary": self.observed_boundary,
            "boundary_callable_identity": self.boundary_callable_identity,
            "baseline_input_digest": self.baseline_input_digest,
            "mutated_input_digest": self.mutated_input_digest,
            "mutation_applied": self.mutation_applied,
            "expected_code": self.expected_code,
            "observed_code": self.observed_code,
            "observed_exception_type": self.observed_exception_type,
            "observed_details_digest": self.observed_details_digest,
            "first_run_evidence_digest": self.first_run_evidence_digest,
            "second_run_evidence_digest": self.second_run_evidence_digest,
            "deterministic_first_failure": self.deterministic_first_failure,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class HFDescriptorAuthorityProof:
    descriptor: HFCompatibilityDescriptor
    checkpoint_descriptor_digest: str
    replay_hf_evidence_digest: str
    report_hf_evidence_digest: str
    dependency_audit_digest: str
    implementation_audit: HFDescriptorGateImplementationAudit
    positive_cases: tuple[HFPositiveProof, ...]
    adversarial_cases: tuple[HFAdversarialResult, ...]
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, HFCompatibilityDescriptor):
            raise TypeError("proof requires HFCompatibilityDescriptor")
        for name in (
            "checkpoint_descriptor_digest",
            "replay_hf_evidence_digest",
            "report_hf_evidence_digest",
            "dependency_audit_digest",
        ):
            _sha(getattr(self, name), name)
        if not isinstance(
            self.implementation_audit, HFDescriptorGateImplementationAudit
        ):
            raise TypeError("proof requires a typed implementation audit")
        if self.implementation_audit.status != "pass":
            raise ValueError("proof requires a clean implementation audit")
        positives, adversarial = (
            tuple(self.positive_cases),
            tuple(self.adversarial_cases),
        )
        if len(adversarial) != ADVERSARIAL_CASE_COUNT:
            raise ValueError("P3.12B.1 requires exactly 77 adversarial cases")
        if (
            len(positives) != POSITIVE_PROOF_COUNT
            or tuple(item.case_id for item in positives) != POSITIVE_CASE_IDS
        ):
            raise ValueError("P3.12B.2 requires the canonical 22 positive proofs")
        if (
            len({item.case_id for item in positives}) != len(positives)
            or len({item.case_id for item in adversarial}) != ADVERSARIAL_CASE_COUNT
        ):
            raise ValueError("proof case IDs must be unique")
        if (
            self.implementation_audit.positive_case_ids != POSITIVE_CASE_IDS
            or tuple(item.case_id for item in adversarial)
            != self.implementation_audit.adversarial_case_ids
        ):
            raise ValueError("proof inventory does not match the implementation audit")
        if any(
            item.outcome not in {"reject", "invariant_preserved"}
            for item in adversarial
        ):
            raise ValueError("proof cannot contain an incomplete adversarial result")
        object.__setattr__(self, "positive_cases", positives)
        object.__setattr__(self, "adversarial_cases", adversarial)
        object.__setattr__(self, "non_claims", tuple(self.non_claims))

    @property
    def evidence_digest(self) -> str:
        return digest(
            {
                "descriptor": self.descriptor.to_dict(),
                "checkpoint_descriptor_digest": self.checkpoint_descriptor_digest,
                "replay_hf_evidence_digest": self.replay_hf_evidence_digest,
                "report_hf_evidence_digest": self.report_hf_evidence_digest,
                "dependency_audit_digest": self.dependency_audit_digest,
                "implementation_audit": self.implementation_audit.to_dict(),
                "positive_cases": [item.to_dict() for item in self.positive_cases],
                "adversarial_cases": [
                    item.to_dict() for item in self.adversarial_cases
                ],
                "non_claims": list(self.non_claims),
            }
        )


def _error_counts(cases: tuple[HFAdversarialResult, ...]) -> dict[str, int]:
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


def build_receipt(proof: HFDescriptorAuthorityProof) -> dict[str, Any]:
    if not isinstance(proof, HFDescriptorAuthorityProof):
        raise TypeError("receipt requires typed executed proof")
    counts = _error_counts(proof.adversarial_cases)
    if any(counts.values()):
        raise ValueError("cannot build a passing receipt from failed case evidence")
    descriptor = proof.descriptor
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "descriptor_schema_version": descriptor.schema_version,
        "descriptor_digest": descriptor.digest,
        "preservation_reference_digest": descriptor.preservation_reference().digest,
        "architecture_id": descriptor.architecture_id,
        "model_type": descriptor.model_type,
        "architecture_config_digest": descriptor.architecture_config_digest,
        "parameter_catalog_digest": descriptor.parameter_catalog_digest,
        "parameter_layout_digest": descriptor.parameter_layout_digest,
        "parameter_projection_digest": descriptor.parameter_projection_digest,
        "architecture_projection_digest": descriptor.architecture_projection.digest,
        "tokenizer_identity_digest": descriptor.tokenizer.digest,
        "vocabulary_identity_digest": descriptor.vocabulary.digest,
        "special_token_identity_digest": descriptor.special_tokens.digest,
        "checkpoint_hf_descriptor_digest": proof.checkpoint_descriptor_digest,
        "replay_hf_evidence_digest": proof.replay_hf_evidence_digest,
        "report_hf_evidence_digest": proof.report_hf_evidence_digest,
        "positive_proof_results": [item.to_dict() for item in proof.positive_cases],
        "adversarial_case_results": [
            item.to_dict() for item in proof.adversarial_cases
        ],
        "positive_proof_count": POSITIVE_PROOF_COUNT,
        "adversarial_case_count": ADVERSARIAL_CASE_COUNT,
        **counts,
        "dependency_audit_digest": proof.dependency_audit_digest,
        "implementation_audit": proof.implementation_audit.to_dict(),
        "implementation_audit_digest": (
            proof.implementation_audit.implementation_audit_digest
        ),
        "non_claims": list(proof.non_claims),
        "evidence_digest": proof.evidence_digest,
    }


def validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = set(
        build_receipt.__annotations__
    )  # retained below as an explicit stable set
    del required
    required = {
        "schema_version",
        "status",
        "descriptor_schema_version",
        "descriptor_digest",
        "preservation_reference_digest",
        "architecture_id",
        "model_type",
        "architecture_config_digest",
        "parameter_catalog_digest",
        "parameter_layout_digest",
        "parameter_projection_digest",
        "architecture_projection_digest",
        "tokenizer_identity_digest",
        "vocabulary_identity_digest",
        "special_token_identity_digest",
        "checkpoint_hf_descriptor_digest",
        "replay_hf_evidence_digest",
        "report_hf_evidence_digest",
        "positive_proof_results",
        "adversarial_case_results",
        "positive_proof_count",
        "adversarial_case_count",
        "mutation_not_applied_count",
        "boundary_mismatch_count",
        "wrong_failure_count",
        "non_deterministic_first_failure_count",
        "unexpected_pass_count",
        "unexpected_failure_count",
        "dependency_audit_digest",
        "implementation_audit",
        "implementation_audit_digest",
        "non_claims",
        "evidence_digest",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("P3.12B receipt fields are missing or unknown")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["status"] != "pass"
        or payload["positive_proof_count"] != POSITIVE_PROOF_COUNT
        or payload["adversarial_case_count"] != ADVERSARIAL_CASE_COUNT
    ):
        raise ValueError("P3.12B receipt schema or status is invalid")
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
        raise ValueError("P3.12B receipt contains failed cases")
    if (
        not isinstance(payload["positive_proof_results"], list)
        or [item.get("case_id") for item in payload["positive_proof_results"]]
        != list(POSITIVE_CASE_IDS)
        or not isinstance(payload["adversarial_case_results"], list)
        or len(payload["adversarial_case_results"]) != ADVERSARIAL_CASE_COUNT
    ):
        raise ValueError("P3.12B receipt adversarial inventory is incomplete")
    implementation_audit = HFDescriptorGateImplementationAudit.from_dict(
        payload["implementation_audit"]
    )
    if (
        implementation_audit.status != "pass"
        or implementation_audit.positive_case_ids != POSITIVE_CASE_IDS
        or implementation_audit.adversarial_case_ids
        != tuple(item["case_id"] for item in payload["adversarial_case_results"])
        or payload["implementation_audit_digest"]
        != implementation_audit.implementation_audit_digest
    ):
        raise ValueError("P3.12B receipt implementation audit is invalid")
    return dict(payload)


__all__ = [
    "ADVERSARIAL_CASE_COUNT",
    "HFAdversarialResult",
    "HFDescriptorAuthorityProof",
    "HFPositiveProof",
    "POSITIVE_CASE_IDS",
    "POSITIVE_PROOF_COUNT",
    "SCHEMA_VERSION",
    "build_receipt",
    "digest",
    "validate_receipt",
]
