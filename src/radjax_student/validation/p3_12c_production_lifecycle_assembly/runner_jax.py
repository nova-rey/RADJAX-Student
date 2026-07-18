"""Executed P3.12C proof over the one production lifecycle assembler.

The literal adversaries live in :mod:`experiments`; observation receives only
the actual invocation.  Expected metadata appears here only after the actual
first failure has been observed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from radjax_student.checkpoints import save_learning_checkpoint_v3
from radjax_student.checkpoints.npz_codec import mapping_pytree_digest
from radjax_student.learning import assemble_jax_learning_lifecycle
from radjax_student.steps import (
    LearningLoopConfig,
    SyntheticBatchSource,
    run_learning_loop,
)
from radjax_student.steps.loop import require_successful_learning_loop
from radjax_student.validation.architecture_audit import (
    build_architecture_audit,
    require_clean_architecture_audit,
)
from radjax_student.validation.p3_11_9_replay.runner_jax import _batch

from . import experiments
from .diagnostic import Invocation, Observation, observe
from .fixtures import fresh_request_and_registries
from .implementation_audit import (
    audit_assembly_authority,
    require_clean_assembly_authority,
)
from .inventory import (
    ADVERSARIAL_CASE_IDS,
    ADVERSARIAL_CASE_SPECS,
    POSITIVE_CASE_IDS,
    AdversarialCaseSpec,
)
from .models import (
    AdversarialResult,
    LifecycleAssemblyProof,
    PositiveResult,
    digest,
)

_EXPERIMENTS: Mapping[str, Callable[[], Invocation]] = {
    "wrong_request_type": experiments.experiment_wrong_request_type,
    "empty_architecture_identity": (experiments.experiment_empty_architecture_identity),
    "empty_objective_identity": experiments.experiment_empty_objective_identity,
    "empty_optimizer_identity": experiments.experiment_empty_optimizer_identity,
    "empty_runtime_identity": experiments.experiment_empty_runtime_identity,
    "executable_plugin_injection_rejected": (
        experiments.experiment_executable_plugin_injection_rejected
    ),
    "unknown_architecture_identity": (
        experiments.experiment_unknown_architecture_identity
    ),
    "architecture_config_identity_mismatch": (
        experiments.experiment_architecture_config_identity_mismatch
    ),
    "architecture_version_mismatch": (
        experiments.experiment_architecture_version_mismatch
    ),
    "architecture_missing_jax_capability": (
        experiments.experiment_architecture_missing_jax_capability
    ),
    "architecture_initialization_incomplete": (
        experiments.experiment_architecture_initialization_incomplete
    ),
    "architecture_hf_descriptor_inconsistent": (
        experiments.experiment_architecture_hf_descriptor_inconsistent
    ),
    "unknown_objective_identity": experiments.experiment_unknown_objective_identity,
    "objective_config_identity_mismatch": (
        experiments.experiment_objective_config_identity_mismatch
    ),
    "objective_version_mismatch": experiments.experiment_objective_version_mismatch,
    "objective_missing_jax_capability": (
        experiments.experiment_objective_missing_jax_capability
    ),
    "objective_surface_unsupported": (
        experiments.experiment_objective_surface_unsupported
    ),
    "objective_surface_not_architecture_derived": (
        experiments.experiment_objective_surface_not_architecture_derived
    ),
    "objective_descriptor_independently_fabricated": (
        experiments.experiment_objective_descriptor_independently_fabricated
    ),
    "unknown_optimizer_identity": experiments.experiment_unknown_optimizer_identity,
    "optimizer_config_identity_mismatch": (
        experiments.experiment_optimizer_config_identity_mismatch
    ),
    "optimizer_missing_jax_capability": (
        experiments.experiment_optimizer_missing_jax_capability
    ),
    "optimizer_state_identity_mismatch": (
        experiments.experiment_optimizer_state_identity_mismatch
    ),
    "optimizer_state_not_backend_initialized": (
        experiments.experiment_optimizer_state_not_backend_initialized
    ),
    "unknown_runtime_identity": experiments.experiment_unknown_runtime_identity,
    "runtime_context_backend_mismatch": (
        experiments.experiment_runtime_context_backend_mismatch
    ),
    "runtime_context_root_seed_mismatch": (
        experiments.experiment_runtime_context_root_seed_mismatch
    ),
    "runtime_key_stream_root_seed_mismatch": (
        experiments.experiment_runtime_key_stream_root_seed_mismatch
    ),
    "runtime_key_stream_not_backend_derived": (
        experiments.experiment_runtime_key_stream_not_backend_derived
    ),
    "lifecycle_component_replacement": (
        experiments.experiment_lifecycle_component_replacement
    ),
    "loop_executor_architecture_replacement": (
        experiments.experiment_loop_executor_architecture_replacement
    ),
    "loop_executor_optimizer_replacement": (
        experiments.experiment_loop_executor_optimizer_replacement
    ),
    "loop_executor_objective_replacement": (
        experiments.experiment_loop_executor_objective_replacement
    ),
    "validation_manual_happy_path_detected": (
        experiments.experiment_validation_manual_happy_path_detected
    ),
    "production_imports_validation_assembly": (
        experiments.experiment_production_imports_validation_assembly
    ),
    "competing_production_assembler_detected": (
        experiments.experiment_competing_production_assembler_detected
    ),
}

assert tuple(_EXPERIMENTS) == ADVERSARIAL_CASE_IDS
assert tuple(item.case_id for item in ADVERSARIAL_CASE_SPECS) == ADVERSARIAL_CASE_IDS


def _callable_identity(invocation: Invocation) -> str:
    return f"{invocation.callable.__module__}.{invocation.callable.__qualname__}"


def _outcome(
    *,
    mutation_applied: bool,
    first: Observation,
    second: Observation,
    intended_boundary: str,
    expected_code: str,
) -> str:
    """Classify only after two observations; this is exact equality only."""

    if not mutation_applied:
        return "mutation_not_applied"
    if first.boundary != intended_boundary:
        return "boundary_mismatch"
    if first.exception_type is None:
        return "unexpected_pass"
    if (
        first.evidence_digest != second.evidence_digest
        or first.boundary != second.boundary
        or first.code != second.code
        or first.exception_type != second.exception_type
    ):
        return "non_deterministic_first_failure"
    if first.code != expected_code:
        return "wrong_failure"
    return "reject"


def _run_adversary(spec: AdversarialCaseSpec) -> AdversarialResult:
    """Run one literal experiment twice from independently fresh inputs."""

    experiment = _EXPERIMENTS[spec.case_id]
    first_invocation = experiment()
    first = observe(first_invocation)
    second_invocation = experiment()
    second = observe(second_invocation)
    first_baseline = digest(first_invocation.baseline_input)
    first_mutated = digest(first_invocation.mutated_input)
    mutation_applied = (
        first_baseline != first_mutated
        and first_baseline == digest(second_invocation.baseline_input)
        and first_mutated == digest(second_invocation.mutated_input)
    )
    outcome = _outcome(
        mutation_applied=mutation_applied,
        first=first,
        second=second,
        intended_boundary=spec.intended_boundary,
        expected_code=spec.expected_code,
    )
    return AdversarialResult(
        case_id=spec.case_id,
        category=spec.category,
        mutation_applied=mutation_applied,
        baseline_input_digest=first_baseline,
        mutated_input_digest=first_mutated,
        intended_boundary=spec.intended_boundary,
        boundary_callable_identity=_callable_identity(first_invocation),
        observed_boundary=first.boundary,
        observed_exception_type=first.exception_type,
        expected_code=spec.expected_code,
        observed_code=first.code,
        deterministic_first_failure=(
            first.evidence_digest == second.evidence_digest
            and first.boundary == second.boundary
            and first.code == second.code
            and first.exception_type == second.exception_type
        ),
        first_run_evidence_digest=first.evidence_digest,
        second_run_evidence_digest=second.evidence_digest,
        outcome=outcome,
    )


def execute_raw_diagnostic() -> tuple[AdversarialResult, ...]:
    """Execute all 36 literal adversaries and retain only actual evidence."""

    return tuple(_run_adversary(spec) for spec in ADVERSARIAL_CASE_SPECS)


def raw_diagnostic_failures() -> tuple[str, ...]:
    """Return only failing case IDs for the required silent-on-success diagnostic."""

    return tuple(
        item.case_id for item in execute_raw_diagnostic() if item.outcome != "reject"
    )


def _tree_digest(directory: Path) -> str:
    return digest(
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
    )


def _finite_mapping(values: Mapping[str, Any]) -> bool:
    return all(math.isfinite(float(value)) for value in values.values())


def _finite_gradients(gradients: Any) -> bool:
    """Check exposed JAX gradient leaves without giving assembly leaf authority."""

    def finite(value: Any) -> bool:
        if isinstance(value, Mapping):
            return bool(value) and all(finite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return bool(value) and all(finite(item) for item in value)
        if hasattr(value, "tolist"):
            return finite(value.tolist())
        return math.isfinite(float(value))

    return finite(gradients)


def _json_evidence(value: Any) -> Any:
    """Normalize typed execution evidence without retaining executable objects."""

    if isinstance(value, Mapping):
        return {str(key): _json_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_evidence(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_evidence(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _execute_product_path(
    root: Path,
) -> tuple[Any, Any, Any, str, str, Mapping[str, Any]]:
    """Execute one generic production step, checkpoint, and compact report."""

    request, registries = fresh_request_and_registries()
    assembled = assemble_jax_learning_lifecycle(request, registries=registries)
    executor = assembled.loop_executor
    before = executor.lifecycle
    before_parameters = mapping_pytree_digest(before.parameters)
    before_carry = mapping_pytree_digest(before.architecture_carry)
    checkpoint_directory = root / "checkpoint"
    saved_checkpoint: list[Any] = []

    def checkpoint(_execution: Any) -> str:
        saved_checkpoint.append(
            save_learning_checkpoint_v3(
                executor.lifecycle.checkpoint(),
                checkpoint_directory,
                optimizer=executor.lifecycle.optimizer,
            )
        )
        return "p312c-checkpoint"

    result = require_successful_learning_loop(
        run_learning_loop(
            config=LearningLoopConfig(max_steps=1, checkpoint_every_n_steps=1),
            architecture=before.architecture,
            architecture_config=before.architecture_config,
            optimizer=before.optimizer,
            optimizer_config=before.optimizer_config,
            optimizer_state=before.optimizer_state,
            learning_state=before.learning_state,
            parameters=before.parameters,
            objective=before.objective_selection,
            batch_source=SyntheticBatchSource((_batch(0),)),
            step_executor=executor,
            checkpoint=checkpoint,
            emit_run_report=True,
            hf_descriptor=before.hf_descriptor,
        )
    )
    after = executor.lifecycle
    execution = result.final_execution
    if execution is None or result.report is None or len(saved_checkpoint) != 1:
        raise RuntimeError("P3.12C product path did not emit complete evidence")
    if not math.isfinite(float(execution.result.loss.loss)):
        raise RuntimeError("P3.12C product path produced a non-finite loss")
    if not (
        _finite_mapping(execution.objective_metrics)
        and _finite_mapping(execution.architecture_metrics)
        and _finite_mapping(execution.optimizer_metrics)
    ):
        raise RuntimeError("P3.12C product path produced non-finite metrics")
    if not _finite_gradients(execution.gradients):
        raise RuntimeError("P3.12C product path produced non-finite gradients")
    if before_parameters == mapping_pytree_digest(after.parameters):
        raise RuntimeError("P3.12C product path did not move parameters")
    if after.optimizer_state.envelope.step != before.optimizer_state.envelope.step + 1:
        raise RuntimeError("P3.12C product path did not advance optimizer state")
    if after.learning_state.global_step != before.learning_state.global_step + 1:
        raise RuntimeError("P3.12C product path did not advance learning state")
    if before_carry == mapping_pytree_digest(after.architecture_carry):
        raise RuntimeError("P3.12C product path did not advance architecture carry")
    restored_request, restored_registries = fresh_request_and_registries()
    restored = assemble_jax_learning_lifecycle(
        restored_request, registries=restored_registries
    ).lifecycle.restore_from_checkpoint(checkpoint_directory)
    if (
        restored.architecture.architecture_id != after.architecture.architecture_id
        or restored.architecture_config != after.architecture_config
        or restored.parameter_catalog != after.parameter_catalog
        or restored.parameter_layout != after.parameter_layout
        or restored.hf_descriptor != after.hf_descriptor
        or restored.hf_reference != after.hf_reference
        or restored.objective_selection != after.objective_selection
        or restored.objective_config != after.objective_config
        or restored.objective_descriptor != after.objective_descriptor
        or restored.optimizer.optimizer_id != after.optimizer.optimizer_id
        or restored.optimizer_config != after.optimizer_config
        or restored.runtime_reference != after.runtime_reference
        or restored.learning_state != after.learning_state
    ):
        raise RuntimeError("P3.12C checkpoint identity did not restore")
    report = result.report
    if (
        report.objective is None
        or report.objective.descriptor != after.objective_descriptor
        or report.hf is None
        or report.hf.descriptor != after.hf_descriptor
    ):
        raise RuntimeError("P3.12C report identity does not match assembled lifecycle")
    evidence = {
        "loss": execution.result.loss.loss,
        "parameter_moved": before_parameters != mapping_pytree_digest(after.parameters),
        "optimizer_step": after.optimizer_state.envelope.step,
        "learning_step": after.learning_state.global_step,
        "runtime_key_coordinate": _json_evidence(
            execution.runtime_result.output_metadata["rng_bridge"]
        ),
        "checkpoint_reference": saved_checkpoint[0].runtime_reference,
        "report_objective": report.objective.to_dict(),
        "report_hf": report.hf.to_dict(),
    }
    return (
        assembled,
        after,
        report,
        _tree_digest(checkpoint_directory),
        digest(report.to_dict()),
        evidence,
    )


def _positive(case_id: str, boundary: str, evidence: Any) -> PositiveResult:
    return PositiveResult(case_id, boundary, digest(evidence))


def _positive_results(
    assembled: Any,
    after: Any,
    report: Any,
    product_evidence: Mapping[str, Any],
    repeated_digest: str,
) -> tuple[PositiveResult, ...]:
    """Literal, ordered 17-positive proof inventory from executed artifacts."""

    lifecycle = assembled.lifecycle
    positives = (
        _positive(
            "request_constructed",
            "learning.assembly.request",
            lifecycle.architecture_config.to_dict(),
        ),
        _positive(
            "registries_bound",
            "learning.assembly.registries",
            {
                "architecture": dict(assembled.architecture_selection),
                "objective": dict(assembled.objective_selection),
                "optimizer": dict(assembled.optimizer_selection),
                "runtime": dict(assembled.runtime_selection),
            },
        ),
        _positive(
            "architecture_registry_selected",
            "architecture.registry",
            dict(assembled.architecture_selection),
        ),
        _positive(
            "architecture_initialized",
            "architecture.initialize",
            {
                "catalog": lifecycle.parameter_catalog.to_dict(),
                "layout": lifecycle.parameter_layout.to_dict(),
            },
        ),
        _positive(
            "hf_identity_derived",
            "architecture.hf",
            {
                "descriptor": lifecycle.hf_descriptor.to_dict(),
                "reference": lifecycle.hf_reference.to_dict(),
            },
        ),
        _positive(
            "objective_registry_selected",
            "objectives.registry",
            dict(assembled.objective_selection),
        ),
        _positive(
            "objective_surface_resolved",
            "architecture.objective_surface",
            lifecycle.resolved_objective_selection.to_dict(),
        ),
        _positive(
            "objective_descriptor_bound",
            "objectives.execution_descriptor",
            lifecycle.objective_descriptor.to_dict(),
        ),
        _positive(
            "optimizer_registry_selected",
            "optimizers.registry",
            dict(assembled.optimizer_selection),
        ),
        _positive(
            "optimizer_state_initialized",
            "optimizers.initialize",
            {
                "descriptor": lifecycle.optimizer_state.descriptor.to_dict(),
                "envelope": lifecycle.optimizer_state.envelope.to_dict(),
            },
        ),
        _positive(
            "runtime_backend_selected",
            "runtime.selection",
            dict(assembled.runtime_selection),
        ),
        _positive(
            "runtime_context_and_keys_bound",
            "runtime.context",
            {
                "context": lifecycle.runtime_context.to_dict(),
                "key_stream": lifecycle.runtime_key_stream.to_dict(),
            },
        ),
        _positive(
            "lifecycle_constructed",
            "steps.JaxLearningLifecycle",
            {
                "architecture": lifecycle.architecture.architecture_id,
                "objective": lifecycle.objective_descriptor.digest,
                "optimizer": lifecycle.optimizer.optimizer_id,
            },
        ),
        _positive(
            "loop_executor_constructed",
            "steps.JaxLoopExecutor",
            {
                "lifecycle_bound": assembled.loop_executor.lifecycle is lifecycle,
                "rng_slot": assembled.loop_executor.rng_slot,
            },
        ),
        _positive(
            "production_step_executed", "steps.run_learning_loop", product_evidence
        ),
        _positive(
            "checkpoint_and_report_emitted",
            "checkpoints.v3+learning.report",
            {"report": report.to_dict(), "post_step": after.learning_state.to_dict()},
        ),
        _positive(
            "repeated_assembly_identity",
            "learning.assembly.digest",
            {"first": assembled.assembly_digest, "second": repeated_digest},
        ),
    )
    if tuple(item.case_id for item in positives) != POSITIVE_CASE_IDS:
        raise RuntimeError("P3.12C positive inventory drifted from canonical authority")
    return positives


def execute_lifecycle_assembly_proof(root: Path) -> LifecycleAssemblyProof:
    """Run products and raw diagnostic before allowing typed evidence creation."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assembled, after, report, checkpoint_digest, report_digest, product_evidence = (
        _execute_product_path(root / "product")
    )
    second_request, second_registries = fresh_request_and_registries()
    repeated = assemble_jax_learning_lifecycle(
        second_request, registries=second_registries
    )
    if assembled.assembly_digest != repeated.assembly_digest:
        raise RuntimeError("P3.12C repeated assembly identity is not deterministic")
    adversaries = execute_raw_diagnostic()
    if any(item.outcome != "reject" for item in adversaries):
        failures = ",".join(
            item.case_id for item in adversaries if item.outcome != "reject"
        )
        raise RuntimeError("P3.12C raw diagnostic failed: " + failures)
    audit = audit_assembly_authority(Path.cwd())
    require_clean_assembly_authority(audit)
    dependency_audit = build_architecture_audit(Path.cwd())
    require_clean_architecture_audit(dependency_audit)
    return LifecycleAssemblyProof(
        assembled.assembly_digest,
        _positive_results(
            assembled,
            after,
            report,
            product_evidence,
            repeated.assembly_digest,
        ),
        adversaries,
        audit,
        checkpoint_digest,
        report_digest,
        digest(dependency_audit),
    )


def execute_in_temporary_directory() -> LifecycleAssemblyProof:
    """Convenience boundary for scripts/tests without persistent evidence writes."""

    with TemporaryDirectory(prefix="radjax-p312c-") as temporary:
        return execute_lifecycle_assembly_proof(Path(temporary))


__all__ = [
    "execute_in_temporary_directory",
    "execute_lifecycle_assembly_proof",
    "execute_raw_diagnostic",
    "raw_diagnostic_failures",
]
