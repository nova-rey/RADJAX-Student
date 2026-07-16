"""Shared final-gate engine used by both the CLI and focused tests."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from radjax_student.validation.p3_11_9_replay.models import StatefulReplayReceipt
from radjax_student.validation.p3_11_10_gate.adversaries import (
    base_positive,
    exception_identity,
    normalized_input,
    normalized_output,
    run_base_adversary,
)
from radjax_student.validation.p3_11_10_gate.documentation import (
    check_closure_documentation,
)
from radjax_student.validation.p3_11_10_gate.inventory import (
    CASES,
    SECTIONS,
    expected_case_ids,
)
from radjax_student.validation.p3_11_10_gate.models import (
    FinalAdversarialGateProof,
    FinalAdversarialGateReceipt,
    GateBlocker,
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
    """Fail closed before any boundary dispatch when the registry drifts."""

    values = tuple(cases)
    known = tuple(CASES)
    if tuple(item.case_id for item in values) != tuple(item.case_id for item in known):
        raise GateInventoryError("p31110_case_inventory_missing_or_ordered_incorrectly")
    if len({item.case_id for item in values}) != len(values):
        raise GateInventoryError("p31110_case_inventory_duplicate")
    for section in SECTIONS:
        actual = tuple(item.case_id for item in values if item.section_id == section)
        if actual != expected_case_ids(section):
            raise GateInventoryError("p31110_case_inventory_wrong_section")
    if {item.section_id for item in values} != set(SECTIONS):
        raise GateInventoryError("p31110_case_inventory_undeclared_section")
    return values


def _run_once(
    case: GateCaseDefinition, repository_root: Path
) -> tuple[str, GateBlocker | None, str]:
    try:
        if case.expected_outcome == "pass":
            if case.execution_class == "jax_executed_boundary":
                from radjax_student.validation.p3_11_10_gate.runner_jax import (
                    execute_positive,
                )

                value = execute_positive(repository_root)
            else:
                value = base_positive(case.execution_class, repository_root)
            return "pass", None, normalized_output(value)
        if case.execution_class == "jax_executed_boundary":
            from radjax_student.validation.p3_11_10_gate.runner_jax import (
                execute_adversary,
            )

            execute_adversary(case.case_id, repository_root)
        else:
            run_base_adversary(case.execution_class, repository_root)
    except Exception as error:  # The engine validates the identity, never treats arbitrary success as a pass.
        detail = exception_identity(error)
        blocker = GateBlocker(
            case.expected_failure or "p3110_missing_expected_failure",
            case.boundary,
            detail,
        )
        return "reject", blocker, normalized_output(detail)
    return "pass", None, normalized_output({"unexpected": "operation_succeeded"})


def execute_case(case: GateCaseDefinition, repository_root: Path) -> GateCaseResult:
    first_outcome, first_blocker, first_output = _run_once(case, repository_root)
    repeated = True
    if case.expected_outcome == "reject":
        second_outcome, second_blocker, _ = _run_once(case, repository_root)
        repeated = (
            first_outcome == second_outcome
            and (first_blocker is None) == (second_blocker is None)
            and (
                first_blocker is None
                or second_blocker is None
                or first_blocker.details == second_blocker.details
            )
        )
    return GateCaseResult(
        definition=case,
        execution_class=case.execution_class,
        observed_outcome=first_outcome,
        observed_failure=first_blocker,
        intended_boundary_reached=first_outcome == case.expected_outcome,
        repeated_first_failure=repeated,
        input_digest=normalized_input(case.case_id, case.execution_class),
        output_digest=first_output,
        non_claims=NON_CLAIMS,
    )


def _artifact_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay_evidence_digest(repository_root: Path) -> str:
    payload = StatefulReplayReceipt.from_json_bytes(
        (repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes()
    )
    return str(payload["evidence_digest"])


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
    sections = tuple(
        GateSectionResult(section, expected_case_ids(section), tuple(grouped[section]))
        for section in SECTIONS
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
        },
        sections=sections,
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
            "cannot emit a passing final receipt from failed case evidence"
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
