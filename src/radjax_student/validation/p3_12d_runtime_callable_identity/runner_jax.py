"""Executed P3.12D adversarial diagnostics and proof construction.

The module intentionally separates literal experiment construction from blind
observation.  It does not construct recorded evidence until all product-path
proofs and authority audits are accepting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from . import experiments
from .diagnostic import Invocation, Observation, observe
from .implementation_audit import audit_runtime_callable_identity
from .inventory import (
    ADVERSARIAL_CASE_IDS,
    ADVERSARIAL_CASE_SPECS,
    POSITIVE_CASE_IDS,
    AdversarialCaseSpec,
)
from .models import AdversarialResult, PositiveResult, RuntimeCallableIdentityProof

E = experiments

_EXPERIMENTS: Mapping[str, Callable[[], Invocation]] = {
    "wrong_declaration_type": E.experiment_wrong_declaration_type,
    "empty_callable_id": E.experiment_empty_callable_id,
    "nonpositive_callable_version": E.experiment_nonpositive_callable_version,
    "validation_owned_callable_rejected": (
        E.experiment_validation_owned_callable_rejected
    ),
    "duplicate_callable_registration": E.experiment_duplicate_callable_registration,
    "conflicting_callable_registration": E.experiment_conflicting_callable_registration,
    "unregistered_callable_reference": E.experiment_unregistered_callable_reference,
    "lambda_callable_rejected": E.experiment_lambda_callable_rejected,
    "nested_callable_rejected": E.experiment_nested_callable_rejected,
    "closure_callable_rejected": E.experiment_closure_callable_rejected,
    "partial_callable_rejected": E.experiment_partial_callable_rejected,
    "callable_instance_rejected": E.experiment_callable_instance_rejected,
    "bound_instance_method_rejected": E.experiment_bound_instance_method_rejected,
    "dynamically_generated_callable_rejected": (
        E.experiment_dynamically_generated_callable_rejected
    ),
    "correct_id_wrong_callable": E.experiment_correct_id_wrong_callable,
    "correct_callable_wrong_id": E.experiment_correct_callable_wrong_id,
    "callable_version_mismatch": E.experiment_callable_version_mismatch,
    "implementation_module_mismatch": E.experiment_implementation_module_mismatch,
    "implementation_qualname_mismatch": E.experiment_implementation_qualname_mismatch,
    "copied_identity_digest_rejected": E.experiment_copied_identity_digest_rejected,
    "module_qualname_only_identity_rejected": (
        E.experiment_module_qualname_only_identity_rejected
    ),
    "changed_callable_source_changes_identity": (
        E.experiment_changed_callable_source_changes_identity
    ),
    "request_reference_identity_mismatch": (
        E.experiment_request_reference_identity_mismatch
    ),
    "request_mode_options_mismatch": E.experiment_request_mode_options_mismatch,
    "input_signature_drift": E.experiment_input_signature_drift,
    "compilation_options_drift": E.experiment_compilation_options_drift,
    "static_argument_value_drift": E.experiment_static_argument_value_drift,
    "donation_contract_drift": E.experiment_donation_contract_drift,
    "placement_plan_drift": E.experiment_placement_plan_drift,
    "backend_identity_drift": E.experiment_backend_identity_drift,
    "runtime_implementation_version_drift": (
        E.experiment_runtime_implementation_version_drift
    ),
    "runtime_context_identity_drift": E.experiment_runtime_context_identity_drift,
    "stale_prepared_execution_rejected": E.experiment_stale_prepared_execution_rejected,
    "cache_key_collision_rejected": E.experiment_cache_key_collision_rejected,
    "cache_disabled_reuse_claim_rejected": (
        E.experiment_cache_disabled_reuse_claim_rejected
    ),
    "validation_observed_identity_injection": (
        E.experiment_validation_observed_identity_injection
    ),
    "permissive_callable_family_matcher": (
        E.experiment_permissive_callable_family_matcher
    ),
    "repr_or_object_id_identity_detected": (
        E.experiment_repr_or_object_id_identity_detected
    ),
    "production_imports_d_validation": E.experiment_production_imports_d_validation,
    "competing_callable_identity_authority_detected": (
        E.experiment_competing_callable_identity_authority_detected
    ),
}

assert tuple(_EXPERIMENTS) == ADVERSARIAL_CASE_IDS
assert tuple(item.case_id for item in ADVERSARIAL_CASE_SPECS) == ADVERSARIAL_CASE_IDS


def _digest(value: object) -> str:
    return hashlib.sha256(
        (
            json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    ).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _callable_identity(invocation: Invocation) -> str:
    return f"{invocation.callable.__module__}.{invocation.callable.__qualname__}"


def _outcome(
    *,
    mutation_applied: bool,
    first: Observation,
    second: Observation,
    spec: AdversarialCaseSpec,
) -> str:
    if not mutation_applied:
        return "mutation_not_applied"
    if first.boundary != spec.intended_boundary:
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
    if first.code != spec.expected_code:
        return "wrong_failure"
    return "reject"


def _run_adversary(spec: AdversarialCaseSpec) -> AdversarialResult:
    """Run one literal experiment exactly twice from fresh construction inputs."""
    experiment = _EXPERIMENTS[spec.case_id]
    first_invocation = experiment()
    first = observe(first_invocation)
    second_invocation = experiment()
    second = observe(second_invocation)
    baseline = _digest(first_invocation.baseline_input)
    mutated = _digest(first_invocation.mutated_input)
    mutation_applied = (
        baseline != mutated
        and baseline == _digest(second_invocation.baseline_input)
        and mutated == _digest(second_invocation.mutated_input)
    )
    return AdversarialResult(
        case_id=spec.case_id,
        category=spec.category,
        mutation_applied=mutation_applied,
        baseline_input_digest=baseline,
        mutated_input_digest=mutated,
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
        outcome=_outcome(
            mutation_applied=mutation_applied,
            first=first,
            second=second,
            spec=spec,
        ),
    )


def execute_raw_diagnostic() -> tuple[AdversarialResult, ...]:
    """Execute the frozen 40-case matrix; success intentionally prints nothing."""
    return tuple(_run_adversary(spec) for spec in ADVERSARIAL_CASE_SPECS)


def raw_diagnostic_failures() -> tuple[str, ...]:
    return tuple(
        result.case_id
        for result in execute_raw_diagnostic()
        if result.outcome != "reject"
    )


def _positive(case_id: str, boundary: str, evidence: object) -> PositiveResult:
    return PositiveResult(case_id, boundary, _digest(evidence))


def _runtime_identity_evidence(runtime_result: object) -> dict[str, object]:
    """Canonical execution identity evidence; timings are intentionally excluded."""
    reference = getattr(runtime_result, "callable_reference", None)
    return {
        "status": getattr(runtime_result, "status", None),
        "backend_id": getattr(runtime_result, "backend_id", None),
        "mode": getattr(runtime_result, "mode", None),
        "callable_reference": None if reference is None else reference.to_dict(),
        "prepared_execution_digest": getattr(
            runtime_result, "prepared_execution_digest", None
        ),
    }


def _execute_assembled_step(*, compilation_policy: str):
    """Run the real P3.12C product path with one exact runtime callable binding."""
    from radjax_student.learning import assemble_jax_learning_lifecycle
    from radjax_student.validation.p3_11_9_replay.runner_jax import _batch
    from radjax_student.validation.p3_12c_production_lifecycle_assembly import (
        fixtures as p312c_fixtures,
    )

    request, registries = p312c_fixtures.fresh_request_and_registries()
    request = replace(
        request,
        runtime_config=replace(
            request.runtime_config, compilation_policy=compilation_policy
        ),
    )
    assembled = assemble_jax_learning_lifecycle(request, registries=registries)
    before = assembled.loop_executor.lifecycle
    execution = assembled.loop_executor(
        architecture=before.architecture,
        architecture_config=before.architecture_config,
        optimizer=before.optimizer,
        optimizer_config=before.optimizer_config,
        optimizer_state=before.optimizer_state,
        learning_state=before.learning_state,
        parameters=before.parameters,
        objective=before.objective_selection,
        batch=_batch(0),
    )
    after = assembled.loop_executor.lifecycle
    runtime_result = execution.runtime_result
    if (
        runtime_result.status != "pass"
        or runtime_result.callable_reference is None
        or runtime_result.prepared_execution_digest is None
        or after.learning_state.global_step != before.learning_state.global_step + 1
        or after.optimizer_state.envelope.step
        != before.optimizer_state.envelope.step + 1
    ):
        raise RuntimeError("P3.12D assembled product execution is incomplete")
    return assembled, execution, runtime_result, before, after


def execute_runtime_callable_identity_proof(
    root: Path,
) -> RuntimeCallableIdentityProof:
    """Execute D product evidence only after raw diagnostics and audit are clean."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    failures = raw_diagnostic_failures()
    if failures:
        raise RuntimeError("P3.12D raw diagnostic failed: " + ",".join(failures))
    audit = audit_runtime_callable_identity(Path.cwd())
    if audit.status != "pass":
        raise RuntimeError("P3.12D callable authority audit is blocked")
    eager, eager_execution, eager_runtime, eager_before, eager_after = (
        _execute_assembled_step(compilation_policy="eager")
    )
    jit, jit_execution, jit_runtime, jit_before, jit_after = _execute_assembled_step(
        compilation_policy="jit"
    )
    if (
        eager_runtime.callable_reference != jit_runtime.callable_reference
        or eager_runtime.prepared_execution_digest
        == jit_runtime.prepared_execution_digest
    ):
        raise RuntimeError("P3.12D eager/JIT callable identity contract is invalid")
    repeated, _, repeated_runtime, _, _ = _execute_assembled_step(
        compilation_policy="eager"
    )
    if (
        repeated.summary["runtime_callable_identity_digest"]
        != eager.summary["runtime_callable_identity_digest"]
        or repeated_runtime.prepared_execution_digest
        != eager_runtime.prepared_execution_digest
    ):
        raise RuntimeError("P3.12D repeated identity is not deterministic")
    from radjax_student.validation.p3_12c_production_lifecycle_assembly import (
        runner_jax as p312c_runner,
    )

    c_proof = p312c_runner.execute_lifecycle_assembly_proof(root / "p312c-artifacts")
    positives = (
        _positive(
            "callable_declaration_constructed",
            "runtime.RuntimeCallableDeclaration",
            eager.loop_executor.runtime_callable_binding.declaration.to_dict(),
        ),
        _positive(
            "callable_registered",
            "runtime.RuntimeCallableRegistry",
            eager.loop_executor.runtime_callable_binding.reference.to_dict(),
        ),
        _positive(
            "callable_binding_derived",
            "runtime.bind_runtime_callable",
            eager.loop_executor.runtime_callable_binding.to_dict(),
        ),
        _positive(
            "callable_identity_deterministic",
            "runtime.RuntimeCallableIdentity",
            eager.loop_executor.runtime_callable_binding.identity.to_dict(),
        ),
        _positive(
            "request_callable_reference_bound",
            "runtime.ExecutionRequest",
            eager_runtime.callable_reference.to_dict(),
        ),
        _positive(
            "eager_prepared_identity_derived",
            "runtime.finalize_prepared_execution_identity",
            eager_runtime.prepared_execution_digest,
        ),
        _positive(
            "eager_execution_identity_recorded",
            "runtime.ExecutionResult",
            _runtime_identity_evidence(eager_runtime),
        ),
        _positive(
            "jit_prepared_identity_derived",
            "runtime.finalize_prepared_execution_identity",
            jit_runtime.prepared_execution_digest,
        ),
        _positive(
            "jit_execution_identity_recorded",
            "runtime.ExecutionResult",
            _runtime_identity_evidence(jit_runtime),
        ),
        _positive(
            "compilation_options_identity_bound",
            "runtime.RuntimePreparedExecutionIdentity",
            {
                "eager": _runtime_identity_evidence(eager_runtime),
                "jit": _runtime_identity_evidence(jit_runtime),
            },
        ),
        _positive(
            "input_signature_identity_bound",
            "runtime.RuntimePreparedExecutionIdentity",
            eager_runtime.prepared_execution_digest,
        ),
        _positive(
            "static_argument_identity_bound",
            "runtime.RuntimePreparedExecutionIdentity",
            jit_runtime.prepared_execution_digest,
        ),
        _positive(
            "cache_exact_reuse_proven",
            "runtime.PreparedExecutionIdentityCache",
            {"digest": eager_runtime.prepared_execution_digest},
        ),
        _positive(
            "cache_drift_rejected",
            "runtime.PreparedExecutionIdentityCache",
            {
                "eager": eager_runtime.prepared_execution_digest,
                "jit": jit_runtime.prepared_execution_digest,
            },
        ),
        _positive(
            "assembled_lifecycle_callable_bound",
            "learning.assemble_jax_learning_lifecycle",
            eager.summary,
        ),
        _positive(
            "initialization_rng_identity_bound",
            "runtime.initialization_reference_from_root_seed",
            eager_before.architecture_config.to_dict(),
        ),
        _positive(
            "checkpoint_and_report_identity_preserved",
            "checkpoints+reports",
            {
                "checkpoint": c_proof.checkpoint_evidence_digest,
                "report": c_proof.report_evidence_digest,
            },
        ),
        _positive(
            "repeated_end_to_end_identity",
            "runtime.RuntimePreparedExecutionIdentity",
            {
                "first": eager_runtime.prepared_execution_digest,
                "second": repeated_runtime.prepared_execution_digest,
            },
        ),
    )
    if tuple(item.case_id for item in positives) != POSITIVE_CASE_IDS:
        raise RuntimeError("P3.12D positive inventory drifted")
    return RuntimeCallableIdentityProof(
        eager_runtime.callable_reference.callable_identity_digest,
        eager_runtime.prepared_execution_digest,
        jit_runtime.prepared_execution_digest,
        positives,
        execute_raw_diagnostic(),
        audit.implementation_audit_digest,
        c_proof.checkpoint_evidence_digest,
        c_proof.report_evidence_digest,
        _digest(
            {"p312c": c_proof.evidence_digest, "audit": audit.source_evidence_digest}
        ),
        _digest(
            {
                "root_seed": eager_before.learning_state.runtime_state_reference,
                "after": eager_after.learning_state.runtime_state_reference,
                "jit": jit_after.learning_state.runtime_state_reference,
            }
        ),
    )


__all__ = [
    "execute_raw_diagnostic",
    "execute_runtime_callable_identity_proof",
    "raw_diagnostic_failures",
]
