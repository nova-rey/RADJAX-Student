"""Strict, JAX-free canonical evidence models for the final gate."""
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
    canonical_json_bytes,
    parse_canonical_json,
)

GATE_SCHEMA = "radjax.p3_11_10_final_adversarial_gate.v1"
GATE_VERSION = "p3_11_10.v1"
_DIGEST = set("0123456789abcdef")
_EXECUTION_CLASSES = {
    "base_executed_boundary",
    "jax_executed_boundary",
    "checkpoint_filesystem_adversary",
    "replay_evidence_adversary",
    "dependency_import_audit",
    "documentation_claim_audit",
}
_OUTCOMES = {"pass", "reject"}


def _strict(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ReplayCanonicalError(
            f"{name} fields differ; missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayCanonicalError(f"{name} must be a nonempty string")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _DIGEST:
        raise ReplayCanonicalError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ReplayCanonicalError(f"{name} must be boolean")
    return value


def _count(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReplayCanonicalError(f"{name} must be a nonnegative integer")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReplayCanonicalError(f"{name} must be a list")
    result = tuple(_string(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ReplayCanonicalError(f"{name} must not contain duplicates")
    return result


def _freeze(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ReplayCanonicalError(f"{name} must contain finite values")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item, name) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, name) for item in value)
    raise ReplayCanonicalError(f"{name} must be JSON-safe")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class GateBlocker:
    code: str
    boundary: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _string(self.code, "blocker code"))
        object.__setattr__(self, "boundary", _string(self.boundary, "boundary"))
        object.__setattr__(self, "details", _freeze(self.details, "blocker details"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "boundary": self.boundary,
            "details": _thaw(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateBlocker:
        _strict(payload, {"code", "boundary", "details"}, "gate blocker")
        return cls(payload["code"], payload["boundary"], payload["details"])


@dataclass(frozen=True)
class GateCaseDefinition:
    case_id: str
    section_id: str
    execution_class: str
    expected_outcome: Literal["pass", "reject"]
    expected_failure: str | None
    boundary: str
    description: str

    def __post_init__(self) -> None:
        for field_name in ("case_id", "section_id", "boundary", "description"):
            object.__setattr__(
                self, field_name, _string(getattr(self, field_name), field_name)
            )
        if self.execution_class not in _EXECUTION_CLASSES:
            raise ReplayCanonicalError("unsupported gate execution class")
        if self.expected_outcome not in _OUTCOMES:
            raise ReplayCanonicalError("unsupported expected outcome")
        if self.expected_outcome == "pass" and self.expected_failure is not None:
            raise ReplayCanonicalError("positive control cannot declare a failure")
        if self.expected_outcome == "reject":
            object.__setattr__(
                self,
                "expected_failure",
                _string(self.expected_failure, "expected_failure"),
            )

    @property
    def identity(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "section_id": self.section_id,
            "execution_class": self.execution_class,
            "expected_outcome": self.expected_outcome,
            "expected_failure": self.expected_failure,
            "boundary": self.boundary,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateCaseDefinition:
        _strict(
            payload,
            {
                "case_id",
                "section_id",
                "execution_class",
                "expected_outcome",
                "expected_failure",
                "boundary",
                "description",
            },
            "gate case definition",
        )
        return cls(**dict(payload))


@dataclass(frozen=True)
class GateCaseResult:
    definition: GateCaseDefinition
    execution_class: str
    observed_outcome: Literal["pass", "reject"]
    observed_failure: GateBlocker | None
    intended_boundary_reached: bool
    repeated_first_failure: bool
    input_digest: str
    output_digest: str
    non_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.execution_class != self.definition.execution_class:
            raise ReplayCanonicalError("case execution class differs from definition")
        if self.observed_outcome not in _OUTCOMES:
            raise ReplayCanonicalError("unsupported observed outcome")
        if self.observed_outcome == "pass" and self.observed_failure is not None:
            raise ReplayCanonicalError("passing case cannot carry a blocker")
        if self.observed_outcome == "reject" and not isinstance(
            self.observed_failure, GateBlocker
        ):
            raise ReplayCanonicalError("rejected case requires a blocker")
        object.__setattr__(
            self,
            "intended_boundary_reached",
            _bool(self.intended_boundary_reached, "intended_boundary_reached"),
        )
        object.__setattr__(
            self,
            "repeated_first_failure",
            _bool(self.repeated_first_failure, "repeated_first_failure"),
        )
        _digest(self.input_digest, "input_digest")
        _digest(self.output_digest, "output_digest")
        object.__setattr__(self, "non_claims", _strings(self.non_claims, "non_claims"))

    @property
    def passed(self) -> bool:
        if self.definition.expected_outcome != self.observed_outcome:
            return False
        if self.definition.expected_outcome == "pass":
            return self.intended_boundary_reached
        blocker = self.observed_failure
        return bool(
            blocker
            and blocker.code == self.definition.expected_failure
            and blocker.boundary == self.definition.boundary
            and self.intended_boundary_reached
            and self.repeated_first_failure
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.to_dict(),
            "definition_digest": self.definition.identity,
            "execution_class": self.execution_class,
            "observed_outcome": self.observed_outcome,
            "observed_failure": None
            if self.observed_failure is None
            else self.observed_failure.to_dict(),
            "intended_boundary_reached": self.intended_boundary_reached,
            "repeated_first_failure": self.repeated_first_failure,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "result_digest": canonical_digest(
                {
                    "definition": self.definition.to_dict(),
                    "observed_outcome": self.observed_outcome,
                    "observed_failure": None
                    if self.observed_failure is None
                    else self.observed_failure.to_dict(),
                    "input_digest": self.input_digest,
                    "output_digest": self.output_digest,
                }
            ),
            "non_claims": list(self.non_claims),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateCaseResult:
        _strict(
            payload,
            {
                "definition",
                "definition_digest",
                "execution_class",
                "observed_outcome",
                "observed_failure",
                "intended_boundary_reached",
                "repeated_first_failure",
                "input_digest",
                "output_digest",
                "result_digest",
                "non_claims",
            },
            "gate case result",
        )
        definition = GateCaseDefinition.from_dict(payload["definition"])
        _digest(payload["definition_digest"], "definition_digest")
        if payload["definition_digest"] != definition.identity:
            raise ReplayCanonicalError("gate case definition digest mismatch")
        observed = payload["observed_failure"]
        blocker = None if observed is None else GateBlocker.from_dict(observed)
        result = cls(
            definition=definition,
            execution_class=payload["execution_class"],
            observed_outcome=payload["observed_outcome"],
            observed_failure=blocker,
            intended_boundary_reached=payload["intended_boundary_reached"],
            repeated_first_failure=payload["repeated_first_failure"],
            input_digest=payload["input_digest"],
            output_digest=payload["output_digest"],
            non_claims=payload["non_claims"],
        )
        expected = result.to_dict()["result_digest"]
        _digest(payload["result_digest"], "result_digest")
        if payload["result_digest"] != expected:
            raise ReplayCanonicalError("gate case result digest mismatch")
        return result


@dataclass(frozen=True)
class GateSectionResult:
    section_id: str
    expected_case_ids: tuple[str, ...]
    cases: tuple[GateCaseResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_id", _string(self.section_id, "section_id"))
        object.__setattr__(
            self,
            "expected_case_ids",
            _strings(self.expected_case_ids, "expected_case_ids"),
        )
        cases = tuple(self.cases)
        if tuple(case.definition.case_id for case in cases) != self.expected_case_ids:
            raise ReplayCanonicalError(
                "section cases do not exactly match inventory order"
            )
        if any(case.definition.section_id != self.section_id for case in cases):
            raise ReplayCanonicalError("case emitted in wrong gate section")
        object.__setattr__(self, "cases", cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "expected_case_count": len(self.expected_case_ids),
            "executed_case_count": len(self.cases),
            "ordered_case_ids": list(self.expected_case_ids),
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateSectionResult:
        _strict(
            payload,
            {
                "section_id",
                "expected_case_count",
                "executed_case_count",
                "ordered_case_ids",
                "cases",
            },
            "gate section result",
        )
        expected = _strings(payload["ordered_case_ids"], "ordered_case_ids")
        _count(payload["expected_case_count"], "expected_case_count")
        _count(payload["executed_case_count"], "executed_case_count")
        cases = tuple(GateCaseResult.from_dict(item) for item in payload["cases"])
        if payload["expected_case_count"] != len(expected) or payload[
            "executed_case_count"
        ] != len(cases):
            raise ReplayCanonicalError("gate section counts are inconsistent")
        return cls(payload["section_id"], expected, cases)


@dataclass(frozen=True)
class FinalAdversarialGateProof:
    baseline_identities: Mapping[str, str]
    sections: tuple[GateSectionResult, ...]
    replay_evidence_digest: str
    dependency_audit_digest: str
    documentation_consistency_digest: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        baseline = dict(self.baseline_identities)
        if not baseline or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in baseline.items()
        ):
            raise ReplayCanonicalError(
                "baseline identities must be a nonempty string mapping"
            )
        object.__setattr__(
            self,
            "baseline_identities",
            MappingProxyType({key: baseline[key] for key in sorted(baseline)}),
        )
        sections = tuple(self.sections)
        section_ids = tuple(section.section_id for section in sections)
        if len(section_ids) != len(set(section_ids)):
            raise ReplayCanonicalError("duplicate gate section")
        object.__setattr__(self, "sections", sections)
        for field_name in (
            "replay_evidence_digest",
            "dependency_audit_digest",
            "documentation_consistency_digest",
        ):
            _digest(getattr(self, field_name), field_name)
        object.__setattr__(self, "non_claims", _strings(self.non_claims, "non_claims"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_identities": dict(self.baseline_identities),
            "sections": [section.to_dict() for section in self.sections],
            "replay_evidence_digest": self.replay_evidence_digest,
            "dependency_audit_digest": self.dependency_audit_digest,
            "documentation_consistency_digest": self.documentation_consistency_digest,
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class FinalAdversarialGateReceipt:
    proof: FinalAdversarialGateProof

    def to_dict(self) -> dict[str, Any]:
        sections = [section.to_dict() for section in self.proof.sections]
        case_models = [
            case for section in self.proof.sections for case in section.cases
        ]
        cases = [case.to_dict() for case in case_models]
        positive = sum(
            case.definition.expected_outcome == "pass" for case in case_models
        )
        adversarial = len(case_models) - positive
        unexpected_pass = sum(
            case.definition.expected_outcome == "reject"
            and case.observed_outcome == "pass"
            for case in case_models
        )
        unexpected_failure = (
            sum(not case.passed for case in case_models) - unexpected_pass
        )
        status = "pass" if not unexpected_pass and not unexpected_failure else "fail"
        payload = {
            "schema_version": GATE_SCHEMA,
            "gate_version": GATE_VERSION,
            "status": status,
            "baseline_identities": dict(self.proof.baseline_identities),
            "sections": sections,
            "ordered_case_ids": [case["definition"]["case_id"] for case in cases],
            "positive_control_count": positive,
            "adversarial_case_count": adversarial,
            "unexpected_pass_count": unexpected_pass,
            "unexpected_failure_count": unexpected_failure,
            "p3_11_9_replay_evidence_digest": self.proof.replay_evidence_digest,
            "dependency_audit_digest": self.proof.dependency_audit_digest,
            "documentation_consistency_digest": self.proof.documentation_consistency_digest,
            "closure_decision": "local_closure_accepted"
            if status == "pass"
            else "local_closure_rejected",
            "non_claims": list(self.proof.non_claims),
        }
        payload["gate_evidence_digest"] = canonical_digest(payload)
        return payload

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes | str) -> Mapping[str, Any]:
        payload = parse_canonical_json(data)
        expected = {
            "schema_version",
            "gate_version",
            "status",
            "baseline_identities",
            "sections",
            "ordered_case_ids",
            "positive_control_count",
            "adversarial_case_count",
            "unexpected_pass_count",
            "unexpected_failure_count",
            "p3_11_9_replay_evidence_digest",
            "dependency_audit_digest",
            "documentation_consistency_digest",
            "closure_decision",
            "non_claims",
            "gate_evidence_digest",
        }
        _strict(payload, expected, "final gate receipt")
        if (
            payload["schema_version"] != GATE_SCHEMA
            or payload["gate_version"] != GATE_VERSION
        ):
            raise ReplayCanonicalError("unsupported final gate receipt schema")
        if payload["status"] not in {"pass", "fail"} or payload[
            "closure_decision"
        ] not in {"local_closure_accepted", "local_closure_rejected"}:
            raise ReplayCanonicalError("invalid final gate receipt status")
        for field_name in (
            "positive_control_count",
            "adversarial_case_count",
            "unexpected_pass_count",
            "unexpected_failure_count",
        ):
            _count(payload[field_name], field_name)
        for field_name in (
            "p3_11_9_replay_evidence_digest",
            "dependency_audit_digest",
            "documentation_consistency_digest",
            "gate_evidence_digest",
        ):
            _digest(payload[field_name], field_name)
        _strings(payload["ordered_case_ids"], "ordered_case_ids")
        _strings(payload["non_claims"], "non_claims")
        if not isinstance(payload["baseline_identities"], Mapping):
            raise ReplayCanonicalError("baseline_identities must be an object")
        sections = tuple(
            GateSectionResult.from_dict(item) for item in payload["sections"]
        )
        case_models = tuple(case for section in sections for case in section.cases)
        case_ids = tuple(case.definition.case_id for case in case_models)
        if tuple(payload["ordered_case_ids"]) != case_ids:
            raise ReplayCanonicalError("final receipt ordered case inventory mismatch")
        # Local import avoids a models/inventory import cycle while binding a
        # recorded receipt to the maintained complete A-K case registry.
        from radjax_student.validation.p3_11_10_gate.inventory import CASES

        if case_ids != tuple(case.case_id for case in CASES):
            raise ReplayCanonicalError(
                "final receipt does not contain the complete case inventory"
            )
        positive = sum(
            case.definition.expected_outcome == "pass" for case in case_models
        )
        adversarial = len(case_models) - positive
        unexpected_pass = sum(
            case.definition.expected_outcome == "reject"
            and case.observed_outcome == "pass"
            for case in case_models
        )
        unexpected_failure = (
            sum(not case.passed for case in case_models) - unexpected_pass
        )
        if (
            payload["positive_control_count"],
            payload["adversarial_case_count"],
            payload["unexpected_pass_count"],
            payload["unexpected_failure_count"],
        ) != (positive, adversarial, unexpected_pass, unexpected_failure):
            raise ReplayCanonicalError(
                "final receipt counts do not match executed case evidence"
            )
        expected_status = (
            "pass" if not unexpected_pass and not unexpected_failure else "fail"
        )
        if payload["status"] != expected_status:
            raise ReplayCanonicalError(
                "final receipt status does not match case evidence"
            )
        expected_decision = (
            "local_closure_accepted"
            if expected_status == "pass"
            else "local_closure_rejected"
        )
        if payload["closure_decision"] != expected_decision:
            raise ReplayCanonicalError(
                "final receipt closure decision does not match status"
            )
        if (
            canonical_digest(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "gate_evidence_digest"
                }
            )
            != payload["gate_evidence_digest"]
        ):
            raise ReplayCanonicalError("final gate evidence digest mismatch")
        if canonical_json_bytes(payload) != (
            data.encode() if isinstance(data, str) else data
        ):
            raise ReplayCanonicalError("final gate receipt is not canonical JSON")
        return MappingProxyType(payload)


__all__ = [
    "FinalAdversarialGateProof",
    "FinalAdversarialGateReceipt",
    "GATE_SCHEMA",
    "GATE_VERSION",
    "GateBlocker",
    "GateCaseDefinition",
    "GateCaseResult",
    "GateSectionResult",
]
