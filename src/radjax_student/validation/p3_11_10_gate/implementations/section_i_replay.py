"""Literal Section I P3.11.9 replay-artifact and verifier experiments."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from radjax_student.validation.p3_11_9_replay.artifact import (
    validate_recorded_replay_artifact,
)
from radjax_student.validation.p3_11_9_replay.canonical import (
    canonical_digest,
    canonical_json_bytes,
    parse_canonical_json,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_memory_experiment,
    public_boundary,
)


def _receipt(root: Path) -> dict[str, Any]:
    return copy.deepcopy(
        parse_canonical_json((root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes())
    )


def _encoded(payload: dict[str, Any]) -> bytes:
    value = copy.deepcopy(payload)
    value["evidence_digest"] = canonical_digest(
        {key: item for key, item in value.items() if key != "evidence_digest"}
    )
    return canonical_json_bytes(value)


@public_boundary("replay_schema_validation")
def _validate(value: bytes) -> Any:
    return validate_recorded_replay_artifact(value)


def _trace(payload: dict[str, Any], mode: str = "eager") -> dict[str, Any]:
    return payload["modes"][mode]["canonical_trace"]


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
        public_input_kind="p3_11_9_recorded_replay_artifact",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=_validate,
        baseline_callable=_validate,
    )


def experiment_i_replay_a_b_exact_and_read_only_recorded_gate(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["non_claims"].append("recorded_gate_read_only")
    return _record(
        context, baseline, mutated, "non_claims", "append_read_only_replay_nonclaim"
    )


def experiment_i_eager_jit_executed_tolerance_comparison(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["non_claims"].append("executed_eager_jit_tolerance_comparison")
    return _record(
        context, baseline, mutated, "non_claims", "append_cross_mode_nonclaim"
    )


def experiment_i_replay_a_b_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["modes"]["eager"]["replay_b_digest"] = "0" * 64
    return _record(
        context,
        baseline,
        mutated,
        "modes.eager.replay_b_digest",
        "replace_replay_b_digest",
    )


def experiment_i_uninterrupted_resumed_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["modes"]["eager"]["resumed_arm_digest"] = mutated["modes"]["eager"][
        "uninterrupted_arm_digest"
    ]
    return _record(
        context,
        baseline,
        mutated,
        "modes.eager.resumed_arm_digest",
        "replace_resumed_arm_digest_with_uninterrupted_identity",
    )


def experiment_i_batch_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][0]["batch_content_digest"] = "2" * 64
    return _record(
        context,
        baseline,
        mutated,
        "steps[0].batch_content_digest",
        "replace_batch_content_digest",
    )


def experiment_i_batch_order_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    steps = _trace(mutated)["steps"]
    steps[0], steps[1] = steps[1], steps[0]
    return _record(context, baseline, mutated, "steps[0:2]", "swap_replay_batch_order")


def experiment_i_lifecycle_identity_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["lifecycle_identity"]["architecture_id"] = "foreign.architecture"
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.lifecycle_identity.architecture_id",
        "replace_lifecycle_architecture_identity",
    )


def experiment_i_checkpoint_boundary_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["checkpoint_boundary"] = 4
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.checkpoint_boundary",
        "replace_checkpoint_boundary",
    )


def experiment_i_stale_checkpoint_manifest_digest(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["checkpoint_manifest_digest"] = "3" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.checkpoint_manifest_digest",
        "replace_checkpoint_manifest_digest",
    )


def experiment_i_parameter_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_parameter_digest"] = "4" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_parameter_digest",
        "replace_parameter_digest",
    )


def experiment_i_carry_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_architecture_carry_digest"] = "5" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_architecture_carry_digest",
        "replace_carry_digest",
    )


def experiment_i_optimizer_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_optimizer_array_digest"] = "6" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_optimizer_array_digest",
        "replace_optimizer_array_digest",
    )


def experiment_i_learning_state_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_learning_state_digest"] = "7" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_learning_state_digest",
        "replace_learning_state_digest",
    )


def experiment_i_hook_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_hook_digest"] = "8" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_hook_digest",
        "replace_hook_digest",
    )


def experiment_i_metric_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["retained_metric_history_digest"] = "9" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.retained_metric_history_digest",
        "replace_metric_history_digest",
    )


def experiment_i_report_digest_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_report_digest"] = "a" * 64
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_report_digest",
        "replace_report_digest",
    )


def experiment_i_rng_coordinate_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][0]["rng"]["invocation_index"] = 9
    return _record(
        context,
        baseline,
        mutated,
        "steps[0].rng.invocation_index",
        "replace_rng_invocation_index",
    )


def experiment_i_runtime_precision_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_runtime"]["precision_policy"] = "bfloat16"
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_runtime.precision_policy",
        "replace_runtime_precision_policy",
    )


def experiment_i_runtime_placement_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_runtime"]["placement_policy"] = "replicated"
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_runtime.placement_policy",
        "replace_runtime_placement_policy",
    )


def experiment_i_runtime_metadata_fields_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_runtime"]["output_metadata_fields"].append(
        "unexpected_field"
    )
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_runtime.output_metadata_fields",
        "append_runtime_metadata_field",
    )


def experiment_i_cross_mode_float_outside_tolerance(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["floating_values_within_tolerance"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.floating_values_within_tolerance",
        "clear_cross_mode_float_tolerance",
    )


def experiment_i_cross_mode_integer_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["integer_values_equal"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.integer_values_equal",
        "clear_cross_mode_integer_equality",
    )


def experiment_i_cross_mode_dtype_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["dtype_shape_equal"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.dtype_shape_equal",
        "clear_cross_mode_dtype_shape_equality",
    )


def experiment_i_cross_mode_shape_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["leaf_count_equal"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.leaf_count_equal",
        "clear_cross_mode_leaf_count_equality",
    )


def experiment_i_cross_mode_tree_structure_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["structure_equal"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.structure_equal",
        "clear_cross_mode_tree_structure_equality",
    )


def experiment_i_cross_mode_metric_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["metric_values_within_tolerance"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.metric_values_within_tolerance",
        "clear_cross_mode_metric_tolerance",
    )


def experiment_i_cross_mode_hook_order_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["hook_sequence_equal"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.hook_sequence_equal",
        "clear_cross_mode_hook_order_equality",
    )


def experiment_i_cross_mode_logical_path_change(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["logical_paths_equal"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.logical_paths_equal",
        "clear_cross_mode_logical_path_equality",
    )


def experiment_i_hardcoded_cross_mode_result(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode"]["runtime_structure_equal"] = False
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode.runtime_structure_equal",
        "insert_unexecuted_cross_mode_result",
    )


def experiment_i_unknown_replay_top_level_field(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["unknown_top_level"] = "unexpected"
    return _record(
        context,
        baseline,
        mutated,
        "unknown_top_level",
        "insert_unknown_replay_top_level_field",
    )


def experiment_i_unknown_replay_nested_runtime_field(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["final_runtime"]["unknown_runtime_field"] = "unexpected"
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.final_runtime.unknown_runtime_field",
        "insert_unknown_nested_runtime_field",
    )


def experiment_i_unknown_replay_nested_lifecycle_field(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["lifecycle_identity"]["unknown_lifecycle_field"] = "unexpected"
    return _record(
        context,
        baseline,
        mutated,
        "canonical_trace.lifecycle_identity.unknown_lifecycle_field",
        "insert_unknown_nested_lifecycle_field",
    )


def experiment_i_missing_replay_nested_rng_field(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    del _trace(mutated)["steps"][0]["rng"]["slot"]
    return _record(
        context, baseline, mutated, "steps[0].rng.slot", "remove_required_rng_slot"
    )


def experiment_i_malformed_replay_tolerance(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["cross_mode_tolerance"]["rtol"] = "nan"
    return _record(
        context,
        baseline,
        mutated,
        "cross_mode_tolerance.rtol",
        "replace_tolerance_with_noncanonical_scalar",
    )


def experiment_i_nonfinite_canonical_scalar(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    _trace(mutated)["steps"][0]["objective_metrics"]["mse"] = "inf"
    return _record(
        context,
        baseline,
        mutated,
        "steps[0].objective_metrics.mse",
        "replace_metric_with_nonfinite_scalar",
    )


def experiment_i_duplicate_nested_json_field(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    raw = (
        _encoded(mutated)
        .decode("utf-8")
        .replace('"schema_version"', '"schema_version":"duplicate","schema_version"', 1)
        .encode("utf-8")
    )
    return execute_memory_experiment(
        context,
        baseline=_encoded(baseline),
        mutated=raw,
        public_input_kind="p3_11_9_recorded_replay_artifact",
        canonical_path="schema_version",
        operation="duplicate_nested_json_key",
        value_summary={"path": "schema_version"},
        public_callable=_validate,
        baseline_callable=_validate,
    )


def experiment_i_stale_evidence_digest(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["evidence_digest"] = "b" * 64
    return execute_memory_experiment(
        context,
        baseline=_encoded(baseline),
        mutated=canonical_json_bytes(mutated),
        public_input_kind="p3_11_9_recorded_replay_artifact",
        canonical_path="evidence_digest",
        operation="replace_replay_evidence_digest",
        value_summary={"digest": "b" * 64},
        public_callable=_validate,
        baseline_callable=_validate,
    )


def experiment_i_handwritten_passing_receipt(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["verifier"] = {"status": "pass", "blockers": ["handwritten"]}
    return _record(
        context,
        baseline,
        mutated,
        "verifier.blockers",
        "insert_handwritten_passing_verifier",
    )


def experiment_i_committed_artifact_regeneration_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _receipt(context.repository_root)
    mutated = _receipt(context.repository_root)
    mutated["modes"]["jit"]["replay_a_digest"] = "c" * 64
    return _record(
        context,
        baseline,
        mutated,
        "modes.jit.replay_a_digest",
        "replace_regenerated_replay_digest",
    )


def experiment_i_check_recorded_writes_maintained_file(
    context: GateExecutionContext,
) -> ExperimentExecution:
    artifact = context.repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json"
    baseline = {
        "before": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "after": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    mutated = {"before": baseline["before"], "after": "d" * 64}

    @public_boundary("replay_schema_validation")
    def read_only(value: dict[str, str]) -> None:
        if value["before"] != value["after"]:
            raise ValueError("recorded replay command modified a maintained file")

    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="maintained_replay_artifact_hash",
        canonical_path="docs/P3_11_9_REPLAY_EVIDENCE.json",
        operation="replace_recorded_artifact_after_hash",
        value_summary={"path": "docs/P3_11_9_REPLAY_EVIDENCE.json"},
        public_callable=read_only,
        baseline_callable=read_only,
    )


SECTION_IMPLEMENTATIONS = {
    "I.positive.replay_a_b_exact_and_read_only_recorded_gate": GateCaseImplementation(
        experiment_i_replay_a_b_exact_and_read_only_recorded_gate
    ),
    "I.positive.eager_jit_executed_tolerance_comparison": GateCaseImplementation(
        experiment_i_eager_jit_executed_tolerance_comparison
    ),
    "I.reject.replay_a_b_mismatch": GateCaseImplementation(
        experiment_i_replay_a_b_mismatch
    ),
    "I.reject.uninterrupted_resumed_mismatch": GateCaseImplementation(
        experiment_i_uninterrupted_resumed_mismatch
    ),
    "I.reject.batch_digest_change": GateCaseImplementation(
        experiment_i_batch_digest_change
    ),
    "I.reject.batch_order_change": GateCaseImplementation(
        experiment_i_batch_order_change
    ),
    "I.reject.lifecycle_identity_change": GateCaseImplementation(
        experiment_i_lifecycle_identity_change
    ),
    "I.reject.checkpoint_boundary_change": GateCaseImplementation(
        experiment_i_checkpoint_boundary_change
    ),
    "I.reject.stale_checkpoint_manifest_digest": GateCaseImplementation(
        experiment_i_stale_checkpoint_manifest_digest
    ),
    "I.reject.parameter_digest_change": GateCaseImplementation(
        experiment_i_parameter_digest_change
    ),
    "I.reject.carry_digest_change": GateCaseImplementation(
        experiment_i_carry_digest_change
    ),
    "I.reject.optimizer_digest_change": GateCaseImplementation(
        experiment_i_optimizer_digest_change
    ),
    "I.reject.learning_state_digest_change": GateCaseImplementation(
        experiment_i_learning_state_digest_change
    ),
    "I.reject.hook_digest_change": GateCaseImplementation(
        experiment_i_hook_digest_change
    ),
    "I.reject.metric_digest_change": GateCaseImplementation(
        experiment_i_metric_digest_change
    ),
    "I.reject.report_digest_change": GateCaseImplementation(
        experiment_i_report_digest_change
    ),
    "I.reject.rng_coordinate_change": GateCaseImplementation(
        experiment_i_rng_coordinate_change
    ),
    "I.reject.runtime_precision_change": GateCaseImplementation(
        experiment_i_runtime_precision_change
    ),
    "I.reject.runtime_placement_change": GateCaseImplementation(
        experiment_i_runtime_placement_change
    ),
    "I.reject.runtime_metadata_fields_change": GateCaseImplementation(
        experiment_i_runtime_metadata_fields_change
    ),
    "I.reject.cross_mode_float_outside_tolerance": GateCaseImplementation(
        experiment_i_cross_mode_float_outside_tolerance
    ),
    "I.reject.cross_mode_integer_change": GateCaseImplementation(
        experiment_i_cross_mode_integer_change
    ),
    "I.reject.cross_mode_dtype_change": GateCaseImplementation(
        experiment_i_cross_mode_dtype_change
    ),
    "I.reject.cross_mode_shape_change": GateCaseImplementation(
        experiment_i_cross_mode_shape_change
    ),
    "I.reject.cross_mode_tree_structure_change": GateCaseImplementation(
        experiment_i_cross_mode_tree_structure_change
    ),
    "I.reject.cross_mode_metric_change": GateCaseImplementation(
        experiment_i_cross_mode_metric_change
    ),
    "I.reject.cross_mode_hook_order_change": GateCaseImplementation(
        experiment_i_cross_mode_hook_order_change
    ),
    "I.reject.cross_mode_logical_path_change": GateCaseImplementation(
        experiment_i_cross_mode_logical_path_change
    ),
    "I.reject.hardcoded_cross_mode_result": GateCaseImplementation(
        experiment_i_hardcoded_cross_mode_result
    ),
    "I.reject.unknown_replay_top_level_field": GateCaseImplementation(
        experiment_i_unknown_replay_top_level_field
    ),
    "I.reject.unknown_replay_nested_runtime_field": GateCaseImplementation(
        experiment_i_unknown_replay_nested_runtime_field
    ),
    "I.reject.unknown_replay_nested_lifecycle_field": GateCaseImplementation(
        experiment_i_unknown_replay_nested_lifecycle_field
    ),
    "I.reject.missing_replay_nested_rng_field": GateCaseImplementation(
        experiment_i_missing_replay_nested_rng_field
    ),
    "I.reject.malformed_replay_tolerance": GateCaseImplementation(
        experiment_i_malformed_replay_tolerance
    ),
    "I.reject.nonfinite_canonical_scalar": GateCaseImplementation(
        experiment_i_nonfinite_canonical_scalar
    ),
    "I.reject.duplicate_nested_json_field": GateCaseImplementation(
        experiment_i_duplicate_nested_json_field
    ),
    "I.reject.stale_evidence_digest": GateCaseImplementation(
        experiment_i_stale_evidence_digest
    ),
    "I.reject.handwritten_passing_receipt": GateCaseImplementation(
        experiment_i_handwritten_passing_receipt
    ),
    "I.reject.committed_artifact_regeneration_mismatch": GateCaseImplementation(
        experiment_i_committed_artifact_regeneration_mismatch
    ),
    "I.reject.check_recorded_writes_maintained_file": GateCaseImplementation(
        experiment_i_check_recorded_writes_maintained_file
    ),
}
