"""Literal Section K maintained-document claim experiments."""

from __future__ import annotations

from pathlib import Path

from radjax_student.validation.p3_11_10_gate.documentation import (
    check_closure_documentation,
    require_closure_documentation,
    write_minimal_maintained_documents,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_directory_experiment,
    public_boundary,
)


def _create_documents(target: Path) -> Path:
    """Create an actual valid document tree independent of receipt text."""

    return write_minimal_maintained_documents(target)


@public_boundary("documentation_validation")
def _validate(root: Path) -> None:
    require_closure_documentation(check_closure_documentation(root))


def _record(
    context: GateExecutionContext,
    baseline: Path,
    mutated: Path,
    path: str,
    operation: str,
) -> ExperimentExecution:
    return execute_directory_experiment(
        context,
        baseline_directory=baseline,
        mutated_directory=mutated,
        public_input_kind="maintained_document_tree",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=_validate,
        baseline_callable=_validate,
    )


def _append(root: Path, relative: str, line: str) -> None:
    path = root / relative
    path.write_text(
        path.read_text(encoding="utf-8") + "\n" + line + "\n", encoding="utf-8"
    )


def experiment_k_maintained_current_status_and_nonclaims_consistent(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _create_documents(context.temporary_root / "docs-baseline")
    mutated = _create_documents(context.temporary_root / "docs-mutated")
    _append(
        mutated,
        "docs/P3_11_10_FINAL_ADVERSARIAL_GATE.md",
        "P3.11 closure evidence is locally maintained.",
    )
    return _record(
        context,
        baseline,
        mutated,
        "docs/P3_11_10_FINAL_ADVERSARIAL_GATE.md",
        "append_nonclaim_preserving_status_note",
    )


def experiment_k_p3117_8_or_9_pending_claim(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _create_documents(context.temporary_root / "pending-baseline")
    mutated = _create_documents(context.temporary_root / "pending-mutated")
    _append(mutated, "README.md", "P3.11.9 pending acceptance")
    return _record(
        context, baseline, mutated, "README.md", "append_p3119_pending_claim"
    )


def experiment_k_p31110_unstarted_after_closure(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _create_documents(context.temporary_root / "unstarted-baseline")
    mutated = _create_documents(context.temporary_root / "unstarted-mutated")
    _append(mutated, "docs/P3_11_10_FINAL_ADVERSARIAL_GATE.md", "P3.11.10 unstarted")
    return _record(
        context,
        baseline,
        mutated,
        "docs/P3_11_10_FINAL_ADVERSARIAL_GATE.md",
        "append_p31110_unstarted_claim",
    )


def experiment_k_phase4_unstarted_claim(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _create_documents(context.temporary_root / "phase4-baseline")
    mutated = _create_documents(context.temporary_root / "phase4-mutated")
    _append(mutated, "docs/ROADMAP.md", "Phase 4 remains unstarted")
    return _record(
        context, baseline, mutated, "docs/ROADMAP.md", "append_phase4_unstarted_claim"
    )


def experiment_k_unsupported_remote_ci_pass_claim(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _create_documents(context.temporary_root / "ci-baseline")
    mutated = _create_documents(context.temporary_root / "ci-mutated")
    _append(mutated, "README.md", "remote CI passed")
    return _record(
        context,
        baseline,
        mutated,
        "README.md",
        "append_unsupported_remote_ci_success_claim",
    )


def experiment_k_production_model_trained_claim(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _create_documents(context.temporary_root / "model-baseline")
    mutated = _create_documents(context.temporary_root / "model-mutated")
    _append(mutated, "docs/P3_11_INTEGRATION_CLOSURE.md", "production model trained")
    return _record(
        context,
        baseline,
        mutated,
        "docs/P3_11_INTEGRATION_CLOSURE.md",
        "append_production_model_training_claim",
    )


def experiment_k_unsupported_cross_environment_replay_claim(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _create_documents(context.temporary_root / "replay-baseline")
    mutated = _create_documents(context.temporary_root / "replay-mutated")
    _append(
        mutated,
        "docs/P3_11_9_DETERMINISTIC_REPLAY.md",
        "cross-environment bitwise replay",
    )
    return _record(
        context,
        baseline,
        mutated,
        "docs/P3_11_9_DETERMINISTIC_REPLAY.md",
        "append_cross_environment_replay_claim",
    )


SECTION_IMPLEMENTATIONS = {
    "K.positive.maintained_current_status_and_nonclaims_consistent": GateCaseImplementation(  # noqa: E501
        experiment_k_maintained_current_status_and_nonclaims_consistent
    ),
    "K.reject.p3117_8_or_9_pending_claim": GateCaseImplementation(
        experiment_k_p3117_8_or_9_pending_claim
    ),
    "K.reject.p31110_unstarted_after_closure": GateCaseImplementation(
        experiment_k_p31110_unstarted_after_closure
    ),
    "K.reject.phase4_unstarted_claim": GateCaseImplementation(
        experiment_k_phase4_unstarted_claim
    ),
    "K.reject.unsupported_remote_ci_pass_claim": GateCaseImplementation(
        experiment_k_unsupported_remote_ci_pass_claim
    ),
    "K.reject.production_model_trained_claim": GateCaseImplementation(
        experiment_k_production_model_trained_claim
    ),
    "K.reject.unsupported_cross_environment_replay_claim": GateCaseImplementation(
        experiment_k_unsupported_cross_environment_replay_claim
    ),
}
