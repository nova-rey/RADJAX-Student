"""Evidence-coupled P3.11.10B final adversarial gate engine."""

from __future__ import annotations

import hashlib
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from radjax_student.validation.p3_11_9_replay.canonical import canonical_digest
from radjax_student.validation.p3_11_9_replay.models import StatefulReplayReceipt
from radjax_student.validation.p3_11_10_gate.documentation import (
    check_closure_documentation,
)
from radjax_student.validation.p3_11_10_gate.implementations import (
    CASE_IMPLEMENTATIONS,
    GateExecutionContext,
    observe_failure,
    validate_implementations,
)
from radjax_student.validation.p3_11_10_gate.inventory import (
    CASES,
    SECTIONS,
    expected_case_ids,
)
from radjax_student.validation.p3_11_10_gate.models import (
    FinalAdversarialGateProof,
    FinalAdversarialGateReceipt,
    GateCaseDefinition,
    GateCaseResult,
    GateMutationEvidence,
    GateSectionResult,
)

NON_CLAIMS = (
    "no_production_architecture",
    "no_tome_payload_consumption",
    "no_distillation",
    "no_hf_export",
    "no_accelerator_scale_training",
    "no_multi_device_proof",
    "no_distributed_training_proof",
    "no_cross_hardware_bitwise_determinism",
    "no_cross_jax_version_bitwise_determinism",
    "no_performance_claim",
    "no_memory_efficiency_claim",
    "no_radlads_parity_claim",
    "no_production_readiness_claim_beyond_foundation_contracts",
)


class GateInventoryError(ValueError):
    pass


def validate_inventory(
    cases: Iterable[GateCaseDefinition] = CASES,
) -> tuple[GateCaseDefinition, ...]:
    values = tuple(cases)
    if tuple(item.case_id for item in values) != tuple(item.case_id for item in CASES):
        raise GateInventoryError("p31110_case_inventory_missing_or_ordered_incorrectly")
    if len({item.case_id for item in values}) != len(values):
        raise GateInventoryError("p31110_case_inventory_duplicate")
    for section in SECTIONS:
        if tuple(
            item.case_id for item in values if item.section_id == section
        ) != expected_case_ids(section):
            raise GateInventoryError("p31110_case_inventory_wrong_section")
    validate_implementations(values)
    return values


def _run_once(case: GateCaseDefinition, repository_root: Path):
    implementation = CASE_IMPLEMENTATIONS[case.case_id]
    with tempfile.TemporaryDirectory(prefix="radjax-p31110b-") as temporary:
        context = GateExecutionContext(repository_root, Path(temporary))
        prepared = implementation.prepare(context)
        probe = implementation.invoke(prepared, context)
        observed = observe_failure(probe)
        outcome = "reject" if observed is not None else "pass"
        output = canonical_digest(
            {
                "return": None if outcome == "reject" else probe.returned_value,
                "trace": probe.trace_digest,
                "observed": None if observed is None else observed.to_dict(),
            }
        )
        return prepared, probe, observed, outcome, output


def _classification(
    case: GateCaseDefinition, prepared, probe, observed, outcome: str, repeated: bool
) -> tuple[str, bool]:
    if (
        prepared.mutation_delta.baseline_input_digest
        == prepared.mutation_delta.mutated_input_digest
    ):
        return "mutation_not_applied", False
    entered = bool(probe.events and probe.events[0] == "invocation_started")
    if case.expected_outcome == "pass":
        return (
            ("expected_pass", True)
            if outcome == "pass" and entered and probe.post_boundary_reached
            else ("unexpected_failure", False)
        )
    if outcome == "pass" or probe.post_boundary_reached:
        return "unexpected_pass", False
    if not entered or observed is None or observed.boundary != case.boundary:
        return "wrong_boundary", False
    if observed.code != case.expected_failure:
        return "wrong_failure", False
    if not repeated:
        return "nondeterministic_failure", False
    return "expected_rejection", True


def execute_case(case: GateCaseDefinition, repository_root: Path) -> GateCaseResult:
    first = _run_once(case, repository_root)
    second = _run_once(case, repository_root)
    prepared, probe, observed, outcome, output = first
    second_prepared, second_probe, second_observed, second_outcome, _ = second
    repeated = (
        outcome == second_outcome
        and (
            None
            if observed is None
            else (observed.code, observed.boundary, observed.exception_type)
        )
        == (
            None
            if second_observed is None
            else (
                second_observed.code,
                second_observed.boundary,
                second_observed.exception_type,
            )
        )
        and tuple(probe.events) == tuple(second_probe.events)
        and prepared.mutation_delta.operation
        == second_prepared.mutation_delta.operation
        and prepared.mutation_delta.canonical_path
        == second_prepared.mutation_delta.canonical_path
    )
    classification, reached = _classification(
        case, prepared, probe, observed, outcome, repeated
    )
    mutation = GateMutationEvidence(
        case_id=case.case_id,
        mutation_kind=prepared.mutation_delta.operation,
        intended_boundary=case.boundary,
        baseline_digest=prepared.mutation_delta.baseline_input_digest,
        mutated_input_digest=prepared.mutation_delta.mutated_input_digest,
        descriptor=canonical_digest(prepared.mutation_delta.to_dict()),
        execution_class=case.execution_class,
        canonical_path=prepared.mutation_delta.canonical_path,
        operation=prepared.mutation_delta.operation,
        delta_digest=prepared.mutation_delta.digest,
    )
    return GateCaseResult(
        definition=case,
        execution_class=case.execution_class,
        observed_outcome=outcome,
        observed_failure=observed,
        intended_boundary_reached=reached,
        repeated_first_failure=repeated,
        input_digest=prepared.mutation_delta.mutated_input_digest,
        output_digest=output,
        non_claims=NON_CLAIMS,
        mutation=mutation,
        implementation_identity=CASE_IMPLEMENTATIONS[case.case_id].identity,
        classification=classification,
        trace_digest=probe.trace_digest,
        repetition_digest=canonical_digest(
            {
                "first": probe.trace_digest,
                "second": second_probe.trace_digest,
                "mutation": prepared.mutation_delta.digest,
            }
        ),
    )


def _artifact_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay_evidence_digest(repository_root: Path) -> str:
    return str(
        StatefulReplayReceipt.from_json_bytes(
            (repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes()
        )["evidence_digest"]
    )


def execute_gate(repository_root: Path) -> FinalAdversarialGateProof:
    cases = validate_inventory()
    results = [execute_case(case, repository_root) for case in cases]
    grouped: dict[str, list[GateCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.definition.section_id].append(result)
    documentation = check_closure_documentation(repository_root)
    if not documentation.ok:
        raise ValueError(
            "P3.11.10 documentation validation failed: "
            + ", ".join(documentation.errors)
        )
    return FinalAdversarialGateProof(
        baseline_identities={
            "p3_11_9_replay_schema": "radjax.p3_11_9_replay_evidence.v1",
            "p3_11_9_replay_artifact": _artifact_digest(
                repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json"
            ),
            "p3_11_8_systems_receipt": _artifact_digest(
                repository_root / "docs/P3_11_8_STATEFUL_SYSTEMS_RECEIPT.json"
            ),
            "implementation_audit_digest": canonical_digest(
                {
                    item.case_id: CASE_IMPLEMENTATIONS[item.case_id].identity
                    for item in cases
                }
            ),
        },
        sections=tuple(
            GateSectionResult(
                section, expected_case_ids(section), tuple(grouped[section])
            )
            for section in SECTIONS
        ),
        replay_evidence_digest=_replay_evidence_digest(repository_root),
        dependency_audit_digest=_artifact_digest(
            repository_root / "docs/P3_5_DEPENDENCY_AUDIT.json"
        ),
        documentation_consistency_digest=documentation.digest,
        non_claims=NON_CLAIMS,
    )


def build_receipt(proof: FinalAdversarialGateProof) -> FinalAdversarialGateReceipt:
    receipt = FinalAdversarialGateReceipt(proof)
    if receipt.to_dict()["status"] != "pass":
        raise ValueError(
            "cannot emit a passing final receipt from failed executed case evidence"
        )
    return receipt


__all__ = [
    "GateInventoryError",
    "NON_CLAIMS",
    "build_receipt",
    "execute_case",
    "execute_gate",
    "validate_inventory",
]
