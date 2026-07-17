"""Literal Section H continuation-identity and resumed-run experiments."""

from __future__ import annotations

import copy
from typing import Any

from radjax_student.validation.p3_11_9_replay.canonical import (
    canonical_digest,
    canonical_json_bytes,
    parse_canonical_json,
)
from radjax_student.validation.p3_11_9_replay.models import StatefulReplayReceipt
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_memory_experiment,
    public_boundary,
)


def _receipt(root: Any) -> dict[str, Any]:
    return copy.deepcopy(
        parse_canonical_json((root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes())
    )


def _encoded(payload: dict[str, Any]) -> bytes:
    value = copy.deepcopy(payload)
    value["evidence_digest"] = canonical_digest(
        {key: item for key, item in value.items() if key != "evidence_digest"}
    )
    return canonical_json_bytes(value)


@public_boundary("resume_replay_validation")
def _parse(value: bytes) -> Any:
    return StatefulReplayReceipt.from_json_bytes(value)


def _trace(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["modes"]["eager"]["canonical_trace"]


def _record(
    context: GateExecutionContext,
    baseline: dict[str, Any],
    mutated: dict[str, Any],
    path: str,
    operation: str,
) -> ExperimentExecution:
    return execute_memory_experiment(
        context,
        baseline=_encoded(baseline),
        mutated=_encoded(mutated),
        public_input_kind="caller_bound_resume_replay_receipt",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=_parse,
        baseline_callable=_parse,
    )


def experiment_h_eager_uninterrupted_resumed_bitwise_equal(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["non_claims"].append("resume_identity_verified")
    return _record(context, baseline, mutated, "non_claims", "append_resume_nonclaim")


def experiment_h_jit_uninterrupted_resumed_bitwise_equal(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["non_claims"].append("jit_resume_identity_verified")
    return _record(
        context, baseline, mutated, "non_claims", "append_jit_resume_nonclaim"
    )


def experiment_h_same_batch_id_different_content(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][3]["batch_content_digest"] = "0" * 64
    return _record(
        context,
        baseline,
        mutated,
        "steps[3].batch_content_digest",
        "replace_same_id_batch_content_digest",
    )


def experiment_h_reordered_batches(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    steps = _trace(mutated)["steps"]
    steps[3], steps[4] = steps[4], steps[3]
    return _record(
        context, baseline, mutated, "steps[3:5]", "swap_remaining_batch_order"
    )


def experiment_h_different_root_seed(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["root_seed"] = 18
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.root_seed",
        "replace_resume_root_seed",
    )


def experiment_h_different_rng_slot(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][3]["rng"]["slot"] = "augmentation"
    return _record(
        context, baseline, mutated, "steps[3].rng.slot", "replace_resume_rng_slot"
    )


def experiment_h_different_objective_id(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][3]["objective"]["objective_id"] = "foreign.objective.v1"
    return _record(
        context,
        baseline,
        mutated,
        "steps[3].objective.objective_id",
        "replace_resume_objective_id",
    )


def experiment_h_different_update_scope(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][3]["resolved_update_scope_identity"] = (
        "foreign-update-scope"
    )
    return _record(
        context,
        baseline,
        mutated,
        "steps[3].resolved_update_scope_identity",
        "replace_resume_update_scope",
    )


def experiment_h_different_learning_rate(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["optimizer_config"]["learning_rate"] = 0.5
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.optimizer_config.learning_rate",
        "replace_resume_learning_rate",
    )


def experiment_h_different_architecture_config(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["architecture_config_digest"] = "1" * 64
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.architecture_config_digest",
        "replace_resume_architecture_config_digest",
    )


def experiment_h_different_hf_identity(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["hf_reference"]["tokenizer_id"] = "foreign-tokenizer"
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.hf_reference.tokenizer_id",
        "replace_resume_hf_tokenizer_identity",
    )


def experiment_h_different_layout(context: GateExecutionContext) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["parameter_layout_digest"] = "2" * 64
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.parameter_layout_digest",
        "replace_resume_layout_digest",
    )


def experiment_h_different_optimizer_schema(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["optimizer_numerical_state_schema_version"] = (
        "foreign_optimizer_state.v1"
    )
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.optimizer_numerical_state_schema_version",
        "replace_resume_optimizer_schema",
    )


def experiment_h_skip_caller_bound_restore(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated).pop("restore_used_caller_identity")
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.restore_used_caller_identity",
        "remove_caller_bound_restore_evidence",
    )


def experiment_h_manual_array_patch_after_restore(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_parameter_digest"] = "3" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_parameter_digest",
        "replace_restored_parameter_digest",
    )


def experiment_h_stale_architecture_state(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["architecture_state_id"] = "stale-architecture-state"
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.architecture_state_id",
        "replace_restored_architecture_state_identity",
    )


def experiment_h_stale_architecture_carry_identity(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["architecture_carry_descriptor"][
        "pytree_descriptor_digest"
    ] = "4" * 64
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.architecture_carry_descriptor.pytree_descriptor_digest",
        "replace_restored_carry_descriptor_digest",
    )


def experiment_h_wrong_global_step(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][3]["counters_before"]["global_step"] = 99
    return _record(
        context,
        baseline,
        mutated,
        "steps[3].counters_before.global_step",
        "replace_resume_global_step",
    )


def experiment_h_wrong_optimizer_step(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][3]["counters_before"]["optimizer_step"] = 99
    return _record(
        context,
        baseline,
        mutated,
        "steps[3].counters_before.optimizer_step",
        "replace_resume_optimizer_step",
    )


def experiment_h_foreign_runtime_reference(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["experiment_identity"]["runtime_reference"] = "foreign-runtime"
    return _record(
        context,
        baseline,
        mutated,
        "experiment_identity.runtime_reference",
        "replace_resume_runtime_reference",
    )


SECTION_IMPLEMENTATIONS = {
    "H.positive.eager_uninterrupted_resumed_bitwise_equal": GateCaseImplementation(
        experiment_h_eager_uninterrupted_resumed_bitwise_equal
    ),
    "H.positive.jit_uninterrupted_resumed_bitwise_equal": GateCaseImplementation(
        experiment_h_jit_uninterrupted_resumed_bitwise_equal
    ),
    "H.reject.same_batch_id_different_content": GateCaseImplementation(
        experiment_h_same_batch_id_different_content
    ),
    "H.reject.reordered_batches": GateCaseImplementation(
        experiment_h_reordered_batches
    ),
    "H.reject.different_root_seed": GateCaseImplementation(
        experiment_h_different_root_seed
    ),
    "H.reject.different_rng_slot": GateCaseImplementation(
        experiment_h_different_rng_slot
    ),
    "H.reject.different_objective_id": GateCaseImplementation(
        experiment_h_different_objective_id
    ),
    "H.reject.different_update_scope": GateCaseImplementation(
        experiment_h_different_update_scope
    ),
    "H.reject.different_learning_rate": GateCaseImplementation(
        experiment_h_different_learning_rate
    ),
    "H.reject.different_architecture_config": GateCaseImplementation(
        experiment_h_different_architecture_config
    ),
    "H.reject.different_hf_identity": GateCaseImplementation(
        experiment_h_different_hf_identity
    ),
    "H.reject.different_layout": GateCaseImplementation(experiment_h_different_layout),
    "H.reject.different_optimizer_schema": GateCaseImplementation(
        experiment_h_different_optimizer_schema
    ),
    "H.reject.skip_caller_bound_restore": GateCaseImplementation(
        experiment_h_skip_caller_bound_restore
    ),
    "H.reject.manual_array_patch_after_restore": GateCaseImplementation(
        experiment_h_manual_array_patch_after_restore
    ),
    "H.reject.stale_architecture_state": GateCaseImplementation(
        experiment_h_stale_architecture_state
    ),
    "H.reject.stale_architecture_carry_identity": GateCaseImplementation(
        experiment_h_stale_architecture_carry_identity
    ),
    "H.reject.wrong_global_step": GateCaseImplementation(
        experiment_h_wrong_global_step
    ),
    "H.reject.wrong_optimizer_step": GateCaseImplementation(
        experiment_h_wrong_optimizer_step
    ),
    "H.reject.foreign_runtime_reference": GateCaseImplementation(
        experiment_h_foreign_runtime_reference
    ),
}
