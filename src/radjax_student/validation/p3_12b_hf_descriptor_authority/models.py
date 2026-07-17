"""Strict JAX-free P3.12B executed-proof and receipt contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from radjax_student.contracts import HFCompatibilityDescriptor

SCHEMA_VERSION = "radjax.p3_12b_hf_descriptor_authority.v1"


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
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class HFProofCase:
    case_id: str
    category: Literal["positive", "adversarial"]
    outcome: Literal["pass", "reject"]
    observed_code: str | None
    boundary: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be nonempty")
        if self.category not in {"positive", "adversarial"}:
            raise ValueError("case category is invalid")
        if self.outcome not in {"pass", "reject"}:
            raise ValueError("case outcome is invalid")
        if (self.outcome == "pass") != (self.observed_code is None):
            raise ValueError("case outcome and observed code disagree")
        if not isinstance(self.boundary, str) or not self.boundary:
            raise ValueError("case boundary is invalid")
        _sha(self.evidence_digest, "case evidence digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "outcome": self.outcome,
            "observed_code": self.observed_code,
            "boundary": self.boundary,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class HFDescriptorAuthorityProof:
    descriptor: HFCompatibilityDescriptor
    checkpoint_descriptor_digest: str
    replay_hf_evidence_digest: str
    report_hf_evidence_digest: str
    dependency_audit_digest: str
    positive_cases: tuple[HFProofCase, ...]
    adversarial_cases: tuple[HFProofCase, ...]
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
        positives, adversarial = (
            tuple(self.positive_cases),
            tuple(self.adversarial_cases),
        )
        if (
            not positives
            or not adversarial
            or any(
                item.category != "positive" or item.outcome != "pass"
                for item in positives
            )
            or any(
                item.category != "adversarial" or item.outcome != "reject"
                for item in adversarial
            )
        ):
            raise ValueError("proof case outcomes are invalid")
        if len({item.case_id for item in positives + adversarial}) != len(
            positives + adversarial
        ):
            raise ValueError("proof case IDs must be unique")
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
                "positive_cases": [item.to_dict() for item in self.positive_cases],
                "adversarial_cases": [
                    item.to_dict() for item in self.adversarial_cases
                ],
                "non_claims": list(self.non_claims),
            }
        )


def build_receipt(proof: HFDescriptorAuthorityProof) -> dict[str, Any]:
    if not isinstance(proof, HFDescriptorAuthorityProof):
        raise TypeError("receipt requires typed executed proof")
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
        "positive_proof_count": len(proof.positive_cases),
        "adversarial_case_count": len(proof.adversarial_cases),
        "unexpected_pass_count": 0,
        "unexpected_failure_count": 0,
        "dependency_audit_digest": proof.dependency_audit_digest,
        "non_claims": list(proof.non_claims),
        "evidence_digest": proof.evidence_digest,
    }


def validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
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
        "unexpected_pass_count",
        "unexpected_failure_count",
        "dependency_audit_digest",
        "non_claims",
        "evidence_digest",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("P3.12B receipt fields are missing or unknown")
    if payload["schema_version"] != SCHEMA_VERSION or payload["status"] != "pass":
        raise ValueError("P3.12B receipt schema or status is invalid")
    for name in required - {
        "schema_version",
        "status",
        "architecture_id",
        "model_type",
        "positive_proof_results",
        "adversarial_case_results",
        "positive_proof_count",
        "adversarial_case_count",
        "unexpected_pass_count",
        "unexpected_failure_count",
        "non_claims",
        "descriptor_schema_version",
    }:
        _sha(payload[name], name)
    if (
        payload["unexpected_pass_count"] != 0
        or payload["unexpected_failure_count"] != 0
    ):
        raise ValueError("P3.12B receipt contains unexpected results")
    return dict(payload)


__all__ = [
    "HFDescriptorAuthorityProof",
    "HFProofCase",
    "SCHEMA_VERSION",
    "build_receipt",
    "digest",
    "validate_receipt",
]
