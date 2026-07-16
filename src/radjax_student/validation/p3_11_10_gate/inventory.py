"""The complete ordered P3.11.10 case inventory.

This is deliberately data, not documentation: the engine refuses to issue a
receipt unless it executes every item below exactly once and in this order.
"""
# ruff: noqa: E501

from __future__ import annotations

from radjax_student.validation.p3_11_10_gate.models import GateCaseDefinition


def _cases(
    section: str,
    execution_class: str,
    positives: tuple[str, ...],
    negatives: tuple[str, ...],
) -> tuple[GateCaseDefinition, ...]:
    values: list[GateCaseDefinition] = []
    for slug in positives:
        values.append(
            GateCaseDefinition(
                f"{section}.positive.{slug}",
                section,
                execution_class,
                "pass",
                None,
                _BOUNDARIES[section],
                slug.replace("_", " "),
            )
        )
    for slug in negatives:
        case_id = f"{section}.reject.{slug}"
        values.append(
            GateCaseDefinition(
                case_id,
                section,
                execution_class,
                "reject",
                _expected_failure(section, slug),
                _BOUNDARIES[section],
                slug.replace("_", " "),
            )
        )
    return tuple(values)


_BOUNDARIES = {
    "A": "registry_validation",
    "B": "parameter_layout_validation",
    "C": "learning_batch_validation",
    "D": "runtime_rng_validation",
    "E": "optimizer_registry_validation",
    "F": "loop_executor_validation",
    "G": "checkpoint_restore_validation",
    "H": "resume_replay_validation",
    "I": "replay_schema_validation",
    "J": "dependency_audit_validation",
    "K": "documentation_validation",
}


def _expected_failure(section: str, slug: str) -> str:
    """Expected identities are inventory data, not copied into observations."""

    if section == "A":
        return (
            "optimizer_config_invalid"
            if slug.startswith("optimizer_")
            else "architecture_plugin_invalid"
        )
    return {
        "B": "type_error",
        "C": "type_error",
        "D": "value_error",
        "E": "optimizer_config_invalid",
        "F": "type_error",
        "G": "checkpoint_component_unreadable",
        "H": "replay_canonical_error",
        "I": "replay_canonical_error",
        "J": "dependency_audit_rejected",
        "K": "documentation_validation_rejected",
    }[section]


_SECTION_CASES = {
    "A": _cases(
        "A",
        "base_executed_boundary",
        ("complete_architecture_and_optimizer_register_and_execute",),
        (
            "architecture_apply_jax_only",
            "architecture_registry_id_mismatch",
            "architecture_capability_profile_id_mismatch",
            "architecture_declared_jax_missing_implementation",
            "architecture_undeclared_jax_implementation",
            "architecture_config_different_id",
            "parameter_catalog_different_architecture_id",
            "parameter_layout_different_architecture_id",
            "hf_reference_different_architecture_id",
            "optimizer_jax_helpers_only",
            "optimizer_registry_id_mismatch",
            "optimizer_capability_profile_id_or_version_mismatch",
            "optimizer_declared_jax_missing_implementation",
            "optimizer_jax_implementation_missing_backend_identity",
            "objective_raw_parameter_tree_assumption",
            "legacy_student_registry_protocol",
            "legacy_jax_step_into_loop_executor",
        ),
    ),
    "B": _cases(
        "B",
        "jax_executed_boundary",
        ("trunk_changes_head_bias_unchanged_and_changed_mask_matches",),
        (
            "missing_parameter_leaf",
            "extra_parameter_leaf",
            "wrong_parameter_keypath",
            "wrong_parameter_shape",
            "wrong_parameter_dtype",
            "catalog_layout_logical_path_mismatch",
            "duplicate_logical_path",
            "duplicate_physical_keypath",
            "malformed_nested_layout_metadata",
            "noncanonical_metadata_key",
            "unknown_update_region",
            "unavailable_objective_surface",
            "explicit_unknown_parameter_path",
            "overlapping_changed_unchanged_evidence",
            "excluded_parameter_changes",
            "excluded_optimizer_state_changes",
            "selected_zero_change_claimed_changed",
            "unselected_parameter_claimed_changed",
        ),
    ),
    "C": _cases(
        "C",
        "jax_executed_boundary",
        ("validated_learning_batch_is_materialized_and_executed",),
        (
            "malformed_learning_batch",
            "nonfinite_batch_value",
            "missing_required_batch_input",
            "missing_required_target",
            "validated_batch_a_executed_batch_b",
            "materializer_foreign_source_digest",
            "objective_id_drift",
            "objective_surface_drift",
            "objective_scope_drift",
            "target_shape_mismatch",
            "nonfinite_objective_metric",
            "undeclared_architecture_surface",
            "objective_consumes_parameters",
        ),
    ),
    "D": _cases(
        "D",
        "jax_executed_boundary",
        ("cpu_eager_jit_runtime_and_rng_receipt",),
        (
            "runtime_context_different_backend",
            "foreign_root_seed_key_stream",
            "same_stream_name_wrong_root_seed",
            "invalid_rng_slot",
            "wrong_rng_invocation_index",
            "wrong_rng_global_step",
            "wrong_rng_micro_step",
            "unsupported_prng_implementation_evidence",
            "fabricated_runtime_receipt",
            "missing_placement_evidence",
            "missing_precision_evidence",
            "architecture_device_selection",
            "optimizer_device_selection",
            "learning_direct_jax_jit",
            "runtime_receipt_unstable_device_identity",
            "eager_receipt_claims_compiled",
            "jit_receipt_claims_uncompiled",
            "disallowed_runtime_fallback",
        ),
    ),
    "E": _cases(
        "E",
        "jax_executed_boundary",
        ("sgd_steps_advance_and_non_sgd_semantics_validate",),
        (
            "optimizer_envelope_id_mismatch",
            "optimizer_descriptor_id_mismatch",
            "optimizer_capability_version_mismatch",
            "optimizer_numerical_schema_mismatch",
            "missing_numerical_state_leaf",
            "extra_numerical_state_leaf",
            "malformed_numerical_state_shape",
            "malformed_numerical_state_dtype",
            "envelope_step_numerical_step_mismatch",
            "learning_optimizer_step_envelope_step_mismatch",
            "optimizer_parameter_paths_layout_mismatch",
            "selected_optimizer_state_fails_advance",
            "excluded_optimizer_state_advances",
            "invalid_schedule_value",
            "nan_learning_rate",
            "negative_learning_rate",
            "optimizer_returns_malformed_numerical_state",
            "optimizer_returns_malformed_parameter_pytree",
            "checkpoint_assumes_sgd_step_keypath",
            "non_sgd_state_rejected_as_sgd",
        ),
    ),
    "F": _cases(
        "F",
        "jax_executed_boundary",
        (
            "normal_loop_deterministic_hooks_metrics_report",
            "uninterrupted_resumed_reports_match_within_mode",
        ),
        (
            "unsupported_gradient_accumulation",
            "batch_exhaustion_before_required_steps",
            "hook_blocker_loop_start",
            "hook_blocker_before_step",
            "hook_blocker_after_step",
            "hook_failure_checkpoint_event",
            "step_executor_exception",
            "checkpoint_callback_exception",
            "malformed_step_execution_result",
            "legacy_partial_step_execution_result",
            "learning_state_fails_advance",
            "global_step_without_optimizer_step",
            "micro_step_violates_no_accumulation",
            "duplicate_hook_event_sequence",
            "reordered_hook_event_sequence",
            "nonfinite_metric",
            "duplicate_metric_identity",
            "malformed_report_input",
            "report_claims_unsupported_execution",
            "report_missing_runtime_or_lifecycle_evidence",
        ),
    ),
    "G": _cases(
        "G",
        "checkpoint_filesystem_adversary",
        (
            "deterministic_v3_writes_and_c_fortran_canonical_identity",
            "v2_read_compatibility_and_v3_continuation_restore",
        ),
        (
            "populated_destination_overwrite",
            "incomplete_staged_checkpoint",
            "missing_manifest",
            "extra_unexpected_file",
            "missing_expected_sidecar",
            "sidecar_hash_mismatch",
            "descriptor_hash_mismatch",
            "manifest_hash_mismatch",
            "malformed_json_descriptor",
            "malformed_npz_member_name",
            "extra_npz_member",
            "missing_npz_member",
            "object_dtype",
            "structured_dtype",
            "wrong_tensor_shape",
            "wrong_tensor_dtype",
            "noncanonical_array_order",
            "optimizer_id_tampering",
            "optimizer_capability_version_tampering",
            "optimizer_schema_tampering",
            "optimizer_envelope_step_tampering",
            "optimizer_numerical_step_tampering",
            "optimizer_parameter_paths_tampering",
            "parameter_layout_tampering",
            "hf_identity_tampering",
            "tokenizer_identity_tampering",
            "vocabulary_size_tampering",
            "special_token_digest_tampering",
            "architecture_config_digest_tampering",
            "parameter_catalog_digest_tampering",
            "architecture_state_identity_tampering",
            "architecture_carry_descriptor_tampering",
            "carry_sidecar_changed_rehashed",
            "optimizer_sidecar_changed_rehashed",
            "runtime_reference_mismatch",
            "caller_expected_identity_mismatch",
            "caller_omits_required_lifecycle_expectations",
            "silent_state_repair",
            "crash_preserves_existing_destination",
            "v2_presented_as_v3",
            "unsupported_future_schema",
        ),
    ),
    "H": _cases(
        "H",
        "jax_executed_boundary",
        (
            "eager_uninterrupted_resumed_bitwise_equal",
            "jit_uninterrupted_resumed_bitwise_equal",
        ),
        (
            "same_batch_id_different_content",
            "reordered_batches",
            "different_root_seed",
            "different_rng_slot",
            "different_objective_id",
            "different_update_scope",
            "different_learning_rate",
            "different_architecture_config",
            "different_hf_identity",
            "different_layout",
            "different_optimizer_schema",
            "skip_caller_bound_restore",
            "manual_array_patch_after_restore",
            "stale_architecture_state",
            "stale_architecture_carry_identity",
            "wrong_global_step",
            "wrong_optimizer_step",
            "foreign_runtime_reference",
        ),
    ),
    "I": _cases(
        "I",
        "replay_evidence_adversary",
        (
            "replay_a_b_exact_and_read_only_recorded_gate",
            "eager_jit_executed_tolerance_comparison",
        ),
        (
            "replay_a_b_mismatch",
            "uninterrupted_resumed_mismatch",
            "batch_digest_change",
            "batch_order_change",
            "lifecycle_identity_change",
            "checkpoint_boundary_change",
            "stale_checkpoint_manifest_digest",
            "parameter_digest_change",
            "carry_digest_change",
            "optimizer_digest_change",
            "learning_state_digest_change",
            "hook_digest_change",
            "metric_digest_change",
            "report_digest_change",
            "rng_coordinate_change",
            "runtime_precision_change",
            "runtime_placement_change",
            "runtime_metadata_fields_change",
            "cross_mode_float_outside_tolerance",
            "cross_mode_integer_change",
            "cross_mode_dtype_change",
            "cross_mode_shape_change",
            "cross_mode_tree_structure_change",
            "cross_mode_metric_change",
            "cross_mode_hook_order_change",
            "cross_mode_logical_path_change",
            "hardcoded_cross_mode_result",
            "unknown_replay_top_level_field",
            "unknown_replay_nested_runtime_field",
            "unknown_replay_nested_lifecycle_field",
            "missing_replay_nested_rng_field",
            "malformed_replay_tolerance",
            "nonfinite_canonical_scalar",
            "duplicate_nested_json_field",
            "stale_evidence_digest",
            "handwritten_passing_receipt",
            "committed_artifact_regeneration_mismatch",
            "check_recorded_writes_maintained_file",
        ),
    ),
    "J": _cases(
        "J",
        "dependency_import_audit",
        ("maintained_dependency_audit_exact_accepted_edges",),
        (
            "architecture_imports_validation",
            "optimizer_imports_validation",
            "runtime_imports_validation",
            "checkpoint_imports_validation",
            "generic_learning_imports_jax_execution",
            "objective_imports_concrete_architecture_parameters",
            "architecture_imports_optimizer_implementation",
            "optimizer_imports_architecture_implementation",
            "runtime_imports_architecture_or_optimizer_implementation",
            "learning_imports_legacy_jax_helper",
            "production_imports_validation_stateful_architecture",
            "production_imports_replay_runner",
            "validation_imported_by_public_package_initialization",
            "students_registry_reintroduced",
            "legacy_exported_current_production_namespace",
            "duplicated_contract_class_alias",
            "dependency_cycle_introduction",
        ),
    ),
    "K": _cases(
        "K",
        "documentation_claim_audit",
        ("maintained_current_status_and_nonclaims_consistent",),
        (
            "p3117_8_or_9_pending_claim",
            "p31110_unstarted_after_closure",
            "phase4_already_begun_claim",
            "unsupported_remote_ci_pass_claim",
            "production_model_trained_claim",
            "unsupported_cross_environment_replay_claim",
        ),
    ),
}

SECTIONS = tuple(_SECTION_CASES)
CASES = tuple(case for section in SECTIONS for case in _SECTION_CASES[section])
CASE_BY_ID = {case.case_id: case for case in CASES}

if len(CASE_BY_ID) != len(CASES):  # import-time maintainer error, never receipt data
    raise RuntimeError("P3.11.10 inventory contains duplicate case IDs")


def expected_case_ids(section_id: str) -> tuple[str, ...]:
    return tuple(case.case_id for case in _SECTION_CASES[section_id])


__all__ = ["CASES", "CASE_BY_ID", "SECTIONS", "expected_case_ids"]
