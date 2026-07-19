"""Evidence-coupled literal P3.11.10C final adversarial gate engine."""

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
    implementation_identity_for,
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
    """Reject incomplete, duplicate, reordered, or incorrectly sectioned input."""

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
    """Run one literal experiment from a fresh temporary public input tree."""

    implementation = CASE_IMPLEMENTATIONS[case.case_id]
    with tempfile.TemporaryDirectory(prefix="radjax-p31110c-") as temporary:
        execution = implementation.function(
            GateExecutionContext(repository_root, Path(temporary))
        )
    probe = execution.probe
    observed = observe_failure(probe)
    outcome = "reject" if observed is not None else "pass"
    output = canonical_digest(
        {
            "outcome": outcome,
            "callable": probe.callable_identity,
            "trace": probe.trace_digest,
            "returned_type": (
                None
                if probe.returned_value is None
                else type(probe.returned_value).__qualname__
            ),
            "observed": None if observed is None else observed.to_dict(),
        }
    )
    return execution, observed, outcome, output


def _repetition_matches(first, second) -> bool:
    first_execution, first_observed, first_outcome, _ = first
    second_execution, second_observed, second_outcome, _ = second
    first_probe = first_execution.probe
    second_probe = second_execution.probe
    return (
        first_outcome == second_outcome
        and (
            None
            if first_observed is None
            else (
                first_observed.code,
                first_observed.boundary,
                first_observed.exception_type,
                first_observed.phase,
                first_observed.message_digest,
            )
        )
        == (
            None
            if second_observed is None
            else (
                second_observed.code,
                second_observed.boundary,
                second_observed.exception_type,
                second_observed.phase,
                second_observed.message_digest,
            )
        )
        and first_execution.mutation_delta.digest
        == second_execution.mutation_delta.digest
        and tuple(first_probe.events) == tuple(second_probe.events)
        and first_probe.boundary == second_probe.boundary
        and first_probe.callable_identity == second_probe.callable_identity
    )


def _classification(
    case: GateCaseDefinition,
    execution,
    observed,
    outcome: str,
    repeated: bool,
) -> tuple[str, bool]:
    """Classify a real outcome without exposing expected data to observers."""

    delta = execution.mutation_delta
    probe = execution.probe
    entered = "intended_boundary_entered" in probe.events
    stopped = "execution_stopped" in probe.events
    if delta.baseline_input_digest == delta.mutated_input_digest:
        return "mutation_not_applied", False
    if case.expected_outcome == "pass":
        if outcome == "pass" and entered and probe.post_boundary_reached:
            return "expected_pass", True
        return "unexpected_failure", False
    if outcome == "pass" or probe.post_boundary_reached:
        return "unexpected_pass", False
    if (
        not entered
        or not stopped
        or observed is None
        or observed.boundary != case.boundary
        or probe.boundary != case.boundary
    ):
        return "wrong_boundary", False
    if observed.code != case.expected_failure:
        return "wrong_failure", False
    if not repeated:
        return "nondeterministic_failure", False
    return "expected_rejection", True


def execute_case(case: GateCaseDefinition, repository_root: Path) -> GateCaseResult:
    """Execute one named experiment twice and retain only canonical evidence."""

    first = _run_once(case, repository_root)
    second = _run_once(case, repository_root)
    execution, observed, outcome, output = first
    repeated = _repetition_matches(first, second)
    classification, reached = _classification(
        case, execution, observed, outcome, repeated
    )
    delta = execution.mutation_delta
    mutation = GateMutationEvidence(
        case_id=case.case_id,
        mutation_kind=delta.operation,
        intended_boundary=case.boundary,
        baseline_digest=delta.baseline_input_digest,
        mutated_input_digest=delta.mutated_input_digest,
        descriptor=canonical_digest(delta.to_dict()),
        execution_class=case.execution_class,
        canonical_path=delta.canonical_path,
        operation=delta.operation,
        delta_digest=delta.digest,
    )
    probe = execution.probe
    return GateCaseResult(
        definition=case,
        execution_class=case.execution_class,
        observed_outcome=outcome,
        observed_failure=observed,
        intended_boundary_reached=reached,
        repeated_first_failure=repeated,
        input_digest=delta.mutated_input_digest,
        output_digest=output,
        non_claims=NON_CLAIMS,
        mutation=mutation,
        implementation_identity=implementation_identity_for(case),
        public_callable_identity=probe.callable_identity,
        observed_source_type=(
            ""
            if probe.observed_exception is None
            else type(probe.observed_exception).__name__
        ),
        classification=classification,
        trace_digest=probe.trace_digest,
        repetition_digest=canonical_digest(
            {
                "first": first[0].probe.trace_digest,
                "second": second[0].probe.trace_digest,
                "first_mutation": first[0].mutation_delta.digest,
                "second_mutation": second[0].mutation_delta.digest,
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


def _assert_distinct_public_experiments(results: Iterable[GateCaseResult]) -> None:
    """Reject two inventory entries with the same concrete public experiment."""

    seen: dict[tuple[str, str, str, str], str] = {}
    mutation_inputs: dict[str, str] = {}
    for result in results:
        mutation = result.mutation
        if mutation is None:
            raise GateInventoryError("p31110_case_mutation_missing")
        # Positive controls have no observed failure.  The public callable is
        # nevertheless concrete evidence for them, so never infer it from a
        # blocker payload.
        public_callable = result.public_callable_identity
        key = (
            public_callable,
            mutation.canonical_path,
            mutation.baseline_digest,
            mutation.mutated_input_digest,
        )
        previous = seen.get(key)
        if previous is not None and previous != result.definition.case_id:
            raise GateInventoryError("p31110_case_public_experiment_not_distinct")
        seen[key] = result.definition.case_id
        previous_mutation = mutation_inputs.get(mutation.mutated_input_digest)
        if (
            previous_mutation is not None
            and previous_mutation != result.definition.case_id
        ):
            raise GateInventoryError("p31110_case_mutated_public_input_not_distinct")
        mutation_inputs[mutation.mutated_input_digest] = result.definition.case_id


def execute_gate(repository_root: Path) -> FinalAdversarialGateProof:
    cases = validate_inventory()
    results = [execute_case(case, repository_root) for case in cases]
    _assert_distinct_public_experiments(results)
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
                [result.to_dict() for result in results]
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
