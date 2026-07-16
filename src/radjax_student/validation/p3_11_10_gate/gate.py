"""Evidence-coupled P3.11.10A gate engine."""
# ruff: noqa: E501

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
    BoundaryTrace,
    GateExecutionContext,
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


def _run_once(
    case: GateCaseDefinition, repository_root: Path
) -> tuple[str, object | None, str, BoundaryTrace, str]:
    implementation = CASE_IMPLEMENTATIONS[case.case_id]
    trace = BoundaryTrace()
    with tempfile.TemporaryDirectory(prefix="radjax-p31110a-") as temporary:
        context = GateExecutionContext(repository_root, Path(temporary), trace)
        prepared = implementation.prepare_case(context, case)
        try:
            value = implementation.invoke(prepared, context)  # type: ignore[call-arg]
        except Exception as error:
            observed = implementation.observe(error, implementation.expected_boundary)
            trace.emit("failure_observed", implementation.expected_boundary)
            trace.emit("execution_stopped", implementation.expected_boundary)
            return (
                "reject",
                observed,
                canonical_digest({"exception": observed.to_dict()}),
                trace,
                prepared.mutation.identity,
            )
        trace.emit("boundary_exited", implementation.expected_boundary)
        return (
            "pass",
            None,
            canonical_digest(value),
            trace,
            prepared.mutation.identity,
        )


def _trace_is_terminal(trace: BoundaryTrace, boundary: str, outcome: str) -> bool:
    events = tuple(trace.events)
    if not events or any(event_boundary != boundary for _, event_boundary in events):
        return False
    phases = tuple(phase for phase, _ in events)
    prefix = ("preparation_completed", "mutation_applied", "intended_boundary_entered")
    if phases[:3] != prefix:
        return False
    terminal = (
        ("failure_observed", "execution_stopped")
        if outcome == "reject"
        else ("boundary_exited",)
    )
    return phases[3:] == terminal


def _classification(
    case: GateCaseDefinition,
    outcome: str,
    observed: object | None,
    trace: BoundaryTrace,
    repeated: bool,
    mutation_identity: str,
) -> tuple[str, bool]:
    implementation = CASE_IMPLEMENTATIONS[case.case_id]
    if not mutation_identity:
        return "mutation_not_applied", False
    if case.expected_outcome == "pass":
        return (
            ("expected_pass", True)
            if outcome == "pass"
            else ("unexpected_failure", False)
        )
    if outcome == "pass":
        return "unexpected_pass", False
    if not repeated:
        return "nondeterministic_failure", False
    if observed is None:
        return "unexpected_failure", False
    if not _trace_is_terminal(trace, implementation.expected_boundary, outcome):
        return "wrong_boundary", False
    if getattr(observed, "boundary", None) != implementation.expected_boundary:
        return "wrong_boundary", False
    if getattr(observed, "code", None) != case.expected_failure:
        return "wrong_failure", False
    return "expected_rejection", True


def execute_case(case: GateCaseDefinition, repository_root: Path) -> GateCaseResult:
    first_outcome, first_observed, first_output, first_trace, mutation_identity = (
        _run_once(case, repository_root)
    )
    second_outcome, second_observed, _second_output, second_trace, second_mutation = (
        _run_once(case, repository_root)
    )
    repeated = (
        first_outcome == second_outcome
        and getattr(first_observed, "code", None)
        == getattr(second_observed, "code", None)
        and getattr(first_observed, "boundary", None)
        == getattr(second_observed, "boundary", None)
        and first_trace.events == second_trace.events
        and mutation_identity == second_mutation
    )
    classification, reached = _classification(
        case, first_outcome, first_observed, first_trace, repeated, mutation_identity
    )
    implementation = CASE_IMPLEMENTATIONS[case.case_id]
    # Rebuild the mutation evidence once for the immutable receipt model.
    with tempfile.TemporaryDirectory(prefix="radjax-p31110a-evidence-") as temporary:
        trace = BoundaryTrace()
        prepared = implementation.prepare_case(
            GateExecutionContext(repository_root, Path(temporary), trace), case
        )
    return GateCaseResult(
        definition=case,
        execution_class=case.execution_class,
        observed_outcome=first_outcome,
        observed_failure=first_observed,  # type: ignore[arg-type]
        intended_boundary_reached=reached,
        repeated_first_failure=repeated,
        input_digest=prepared.mutation.mutated_input_digest,
        output_digest=first_output,
        non_claims=NON_CLAIMS,
        mutation=prepared.mutation,
        implementation_identity=implementation.identity,
        classification=classification,
        trace_digest=first_trace.digest,
        repetition_digest=canonical_digest(
            {
                "first": first_trace.digest,
                "second": second_trace.digest,
                "mutation": mutation_identity,
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
    documentation = check_closure_documentation(repository_root)
    if not documentation.ok:
        raise ValueError(
            "P3.11.10 documentation validation failed: "
            + ", ".join(documentation.errors)
        )
    results = [execute_case(case, repository_root) for case in cases]
    grouped: dict[str, list[GateCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.definition.section_id].append(result)
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
