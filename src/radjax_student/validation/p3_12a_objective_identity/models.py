"""Strict, JAX-free receipt contracts for the P3.12A identity proof."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from radjax_student.contracts import ObjectiveExecutionDescriptor, ObjectiveIdentity

SCHEMA_VERSION = "radjax.p3_12a_objective_identity.v1"
_DIGEST = set("0123456789abcdef")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ObjectiveProofCase:
    case_id: str
    category: Literal["positive", "adversarial"]
    outcome: Literal["pass", "reject"]
    expected_code: str | None
    observed_code: str | None
    boundary: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("objective proof case ID must be nonempty")
        if self.category not in {"positive", "adversarial"}:
            raise ValueError("objective proof case category is invalid")
        if self.outcome not in {"pass", "reject"}:
            raise ValueError("objective proof case outcome is invalid")
        if (self.outcome == "pass") != (self.observed_code is None):
            raise ValueError("objective proof outcome and observed code disagree")
        if (self.outcome == "pass") != (self.expected_code is None):
            raise ValueError("objective proof outcome and expected code disagree")
        if self.outcome == "reject" and self.expected_code != self.observed_code:
            raise ValueError("objective proof rejection did not reach expected code")
        if not isinstance(self.boundary, str) or not self.boundary:
            raise ValueError("objective proof boundary is invalid")
        if self.observed_code is not None and (
            not isinstance(self.observed_code, str) or not self.observed_code
        ):
            raise ValueError("objective proof observed code is invalid")
        _sha(self.evidence_digest, "objective proof evidence digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "outcome": self.outcome,
            "expected_code": self.expected_code,
            "observed_code": self.observed_code,
            "boundary": self.boundary,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveProofCase:
        expected = {
            "case_id",
            "category",
            "outcome",
            "expected_code",
            "observed_code",
            "boundary",
            "evidence_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("objective proof case fields are missing or unknown")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ObjectiveIdentityProof:
    descriptor: ObjectiveExecutionDescriptor
    positive_cases: tuple[ObjectiveProofCase, ...]
    adversarial_cases: tuple[ObjectiveProofCase, ...]
    checkpoint_objective_identity_digest: str
    replay_objective_evidence_digest: str
    report_objective_evidence_digest: str
    dependency_audit_digest: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ObjectiveExecutionDescriptor):
            raise TypeError("objective proof requires ObjectiveExecutionDescriptor")
        for name in (
            "checkpoint_objective_identity_digest",
            "replay_objective_evidence_digest",
            "report_objective_evidence_digest",
            "dependency_audit_digest",
        ):
            _sha(getattr(self, name), name)
        positives = tuple(self.positive_cases)
        adversarial = tuple(self.adversarial_cases)
        if not positives or not adversarial:
            raise ValueError("objective proof requires positive and adversarial cases")
        if any(case.category != "positive" for case in positives) or any(
            case.category != "adversarial" for case in adversarial
        ):
            raise ValueError("objective proof case categories disagree")
        if len({case.case_id for case in positives + adversarial}) != len(
            positives + adversarial
        ):
            raise ValueError("objective proof case IDs must be unique")
        if any(case.outcome != "pass" for case in positives) or any(
            case.outcome != "reject" for case in adversarial
        ):
            raise ValueError("objective proof includes an unexpected case outcome")
        if not all(isinstance(item, str) and item for item in self.non_claims):
            raise ValueError("objective proof non-claims are invalid")
        object.__setattr__(self, "positive_cases", positives)
        object.__setattr__(self, "adversarial_cases", adversarial)
        object.__setattr__(self, "non_claims", tuple(self.non_claims))

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "positive_cases": [item.to_dict() for item in self.positive_cases],
            "adversarial_cases": [item.to_dict() for item in self.adversarial_cases],
            "checkpoint_objective_identity_digest": (
                self.checkpoint_objective_identity_digest
            ),
            "replay_objective_evidence_digest": self.replay_objective_evidence_digest,
            "report_objective_evidence_digest": self.report_objective_evidence_digest,
            "dependency_audit_digest": self.dependency_audit_digest,
            "non_claims": list(self.non_claims),
        }

    @property
    def evidence_digest(self) -> str:
        return digest(self.to_dict())


def build_receipt(proof: ObjectiveIdentityProof) -> dict[str, Any]:
    """Build a passing receipt only from executed typed proof results."""

    if not isinstance(proof, ObjectiveIdentityProof):
        raise TypeError("objective receipt requires typed executed proof")
    positive = list(proof.positive_cases)
    adversarial = list(proof.adversarial_cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "objective_identity": proof.descriptor.identity.to_dict(),
        "capability_profile_digest": proof.descriptor.capability_profile_digest,
        "objective_config_digest": proof.descriptor.config_digest,
        "metric_schema_identity": proof.descriptor.metric_schema_id,
        "implementation_identity": proof.descriptor.implementation_identity,
        "lifecycle_descriptor_digest": proof.descriptor.digest,
        "checkpoint_objective_identity_digest": (
            proof.checkpoint_objective_identity_digest
        ),
        "replay_objective_evidence_digest": proof.replay_objective_evidence_digest,
        "report_objective_evidence_digest": proof.report_objective_evidence_digest,
        "positive_proof_results": [item.to_dict() for item in positive],
        "adversarial_case_results": [item.to_dict() for item in adversarial],
        "positive_proof_count": len(positive),
        "adversarial_case_count": len(adversarial),
        "unexpected_pass_count": sum(item.outcome == "pass" for item in adversarial),
        "unexpected_failure_count": sum(
            item.outcome != "reject" for item in adversarial
        ),
        "dependency_audit_digest": proof.dependency_audit_digest,
        "non_claims": list(proof.non_claims),
        "evidence_digest": proof.evidence_digest,
    }


def validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly parse an emitted P3.12A receipt without importing JAX."""

    fields = {
        "schema_version",
        "status",
        "objective_identity",
        "capability_profile_digest",
        "objective_config_digest",
        "metric_schema_identity",
        "implementation_identity",
        "lifecycle_descriptor_digest",
        "checkpoint_objective_identity_digest",
        "replay_objective_evidence_digest",
        "report_objective_evidence_digest",
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
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("objective receipt fields are missing or unknown")
    if payload["schema_version"] != SCHEMA_VERSION or payload["status"] != "pass":
        raise ValueError("objective receipt schema or status is invalid")
    identity = ObjectiveIdentity.from_dict(payload["objective_identity"])
    for name in (
        "capability_profile_digest",
        "objective_config_digest",
        "lifecycle_descriptor_digest",
        "checkpoint_objective_identity_digest",
        "replay_objective_evidence_digest",
        "report_objective_evidence_digest",
        "dependency_audit_digest",
        "evidence_digest",
    ):
        _sha(payload[name], name)
    if not isinstance(payload["metric_schema_identity"], str) or not isinstance(
        payload["implementation_identity"], str
    ):
        raise ValueError("objective receipt identity fields are invalid")
    positives = tuple(
        ObjectiveProofCase.from_dict(item) for item in payload["positive_proof_results"]
    )
    adversarial = tuple(
        ObjectiveProofCase.from_dict(item)
        for item in payload["adversarial_case_results"]
    )
    if (
        payload["positive_proof_count"] != len(positives)
        or payload["adversarial_case_count"] != len(adversarial)
        or any(
            item.category != "positive" or item.outcome != "pass" for item in positives
        )
        or any(
            item.category != "adversarial" or item.outcome != "reject"
            for item in adversarial
        )
        or payload["unexpected_pass_count"] != 0
        or payload["unexpected_failure_count"] != 0
    ):
        raise ValueError("objective receipt case results are inconsistent")
    value = dict(payload)
    value["objective_identity"] = identity.to_dict()
    return value


__all__ = [
    "ObjectiveIdentityProof",
    "ObjectiveProofCase",
    "SCHEMA_VERSION",
    "build_receipt",
    "digest",
    "validate_receipt",
]
