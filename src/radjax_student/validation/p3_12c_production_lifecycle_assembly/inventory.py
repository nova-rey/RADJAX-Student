"""Frozen P3.12C inventory authority shared by proof and source audit."""

from __future__ import annotations

from dataclasses import dataclass

POSITIVE_CASE_IDS: tuple[str, ...] = (
    "request_constructed",
    "registries_bound",
    "architecture_registry_selected",
    "architecture_initialized",
    "hf_identity_derived",
    "objective_registry_selected",
    "objective_surface_resolved",
    "objective_descriptor_bound",
    "optimizer_registry_selected",
    "optimizer_state_initialized",
    "runtime_backend_selected",
    "runtime_context_and_keys_bound",
    "lifecycle_constructed",
    "loop_executor_constructed",
    "production_step_executed",
    "checkpoint_and_report_emitted",
    "repeated_assembly_identity",
)

ADVERSARIAL_CASE_IDS: tuple[str, ...] = (
    "wrong_request_type",
    "empty_architecture_identity",
    "empty_objective_identity",
    "empty_optimizer_identity",
    "empty_runtime_identity",
    "executable_plugin_injection_rejected",
    "unknown_architecture_identity",
    "architecture_config_identity_mismatch",
    "architecture_version_mismatch",
    "architecture_missing_jax_capability",
    "architecture_initialization_incomplete",
    "architecture_hf_descriptor_inconsistent",
    "unknown_objective_identity",
    "objective_config_identity_mismatch",
    "objective_version_mismatch",
    "objective_missing_jax_capability",
    "objective_surface_unsupported",
    "objective_surface_not_architecture_derived",
    "objective_descriptor_independently_fabricated",
    "unknown_optimizer_identity",
    "optimizer_config_identity_mismatch",
    "optimizer_missing_jax_capability",
    "optimizer_state_identity_mismatch",
    "optimizer_state_not_backend_initialized",
    "unknown_runtime_identity",
    "runtime_context_backend_mismatch",
    "runtime_context_root_seed_mismatch",
    "runtime_key_stream_root_seed_mismatch",
    "runtime_key_stream_not_backend_derived",
    "lifecycle_component_replacement",
    "loop_executor_architecture_replacement",
    "loop_executor_optimizer_replacement",
    "loop_executor_objective_replacement",
    "validation_manual_happy_path_detected",
    "production_imports_validation_assembly",
    "competing_production_assembler_detected",
)

assert len(POSITIVE_CASE_IDS) == 17
assert len(ADVERSARIAL_CASE_IDS) == 36


@dataclass(frozen=True)
class AdversarialCaseSpec:
    """Expected metadata is held outside actual observation and classification."""

    case_id: str
    category: str
    intended_boundary: str
    expected_code: str


_ASSEMBLER = "radjax_student.learning.assembly.assemble_jax_learning_lifecycle"
_AUDIT = (
    "radjax_student.validation.p3_12c_production_lifecycle_assembly."
    "implementation_audit.require_clean_synthetic_source"
)

ADVERSARIAL_CASE_SPECS: tuple[AdversarialCaseSpec, ...] = (
    AdversarialCaseSpec(
        "wrong_request_type", "request", _ASSEMBLER, "assembly_request_invalid"
    ),
    AdversarialCaseSpec(
        "empty_architecture_identity",
        "request",
        _ASSEMBLER,
        "assembly_architecture_invalid",
    ),
    AdversarialCaseSpec(
        "empty_objective_identity", "request", _ASSEMBLER, "assembly_objective_invalid"
    ),
    AdversarialCaseSpec(
        "empty_optimizer_identity", "request", _ASSEMBLER, "assembly_optimizer_invalid"
    ),
    AdversarialCaseSpec(
        "empty_runtime_identity", "request", _ASSEMBLER, "assembly_runtime_invalid"
    ),
    AdversarialCaseSpec(
        "executable_plugin_injection_rejected",
        "request",
        _ASSEMBLER,
        "assembly_architecture_invalid",
    ),
    AdversarialCaseSpec(
        "unknown_architecture_identity",
        "architecture",
        _ASSEMBLER,
        "assembly_architecture_unknown",
    ),
    AdversarialCaseSpec(
        "architecture_config_identity_mismatch",
        "architecture",
        _ASSEMBLER,
        "assembly_architecture_invalid",
    ),
    AdversarialCaseSpec(
        "architecture_version_mismatch",
        "architecture",
        _ASSEMBLER,
        "assembly_architecture_invalid",
    ),
    AdversarialCaseSpec(
        "architecture_missing_jax_capability",
        "architecture",
        _ASSEMBLER,
        "assembly_architecture_invalid",
    ),
    AdversarialCaseSpec(
        "architecture_initialization_incomplete",
        "architecture",
        _ASSEMBLER,
        "assembly_architecture_result_invalid",
    ),
    AdversarialCaseSpec(
        "architecture_hf_descriptor_inconsistent",
        "architecture",
        _ASSEMBLER,
        "assembly_architecture_result_invalid",
    ),
    AdversarialCaseSpec(
        "unknown_objective_identity",
        "objective",
        _ASSEMBLER,
        "assembly_objective_unknown",
    ),
    AdversarialCaseSpec(
        "objective_config_identity_mismatch",
        "objective",
        _ASSEMBLER,
        "assembly_objective_invalid",
    ),
    AdversarialCaseSpec(
        "objective_version_mismatch",
        "objective",
        _ASSEMBLER,
        "assembly_objective_unknown",
    ),
    AdversarialCaseSpec(
        "objective_missing_jax_capability",
        "objective",
        _ASSEMBLER,
        "assembly_objective_invalid",
    ),
    AdversarialCaseSpec(
        "objective_surface_unsupported",
        "objective",
        _ASSEMBLER,
        "assembly_objective_surface_unsupported",
    ),
    AdversarialCaseSpec(
        "objective_surface_not_architecture_derived",
        "objective",
        _AUDIT,
        "assembly_audit_fixture_surface_fabrication",
    ),
    AdversarialCaseSpec(
        "objective_descriptor_independently_fabricated",
        "objective",
        _ASSEMBLER,
        "assembly_objective_descriptor_invalid",
    ),
    AdversarialCaseSpec(
        "unknown_optimizer_identity",
        "optimizer",
        _ASSEMBLER,
        "assembly_optimizer_unknown",
    ),
    AdversarialCaseSpec(
        "optimizer_config_identity_mismatch",
        "optimizer",
        _ASSEMBLER,
        "assembly_optimizer_invalid",
    ),
    AdversarialCaseSpec(
        "optimizer_missing_jax_capability",
        "optimizer",
        _ASSEMBLER,
        "assembly_optimizer_invalid",
    ),
    AdversarialCaseSpec(
        "optimizer_state_identity_mismatch",
        "optimizer",
        _ASSEMBLER,
        "assembly_optimizer_initialization_failed",
    ),
    AdversarialCaseSpec(
        "optimizer_state_not_backend_initialized",
        "optimizer",
        _ASSEMBLER,
        "assembly_optimizer_initialization_failed",
    ),
    AdversarialCaseSpec(
        "unknown_runtime_identity", "runtime", _ASSEMBLER, "assembly_runtime_unknown"
    ),
    AdversarialCaseSpec(
        "runtime_context_backend_mismatch",
        "runtime",
        _ASSEMBLER,
        "assembly_runtime_context_mismatch",
    ),
    AdversarialCaseSpec(
        "runtime_context_root_seed_mismatch",
        "runtime",
        _ASSEMBLER,
        "assembly_runtime_context_mismatch",
    ),
    AdversarialCaseSpec(
        "runtime_key_stream_root_seed_mismatch",
        "runtime",
        _ASSEMBLER,
        "assembly_runtime_key_stream_mismatch",
    ),
    AdversarialCaseSpec(
        "runtime_key_stream_not_backend_derived",
        "runtime",
        _AUDIT,
        "assembly_audit_fixture_runtime_key_injection",
    ),
    AdversarialCaseSpec(
        "lifecycle_component_replacement",
        "authority",
        _AUDIT,
        "assembly_audit_fixture_lifecycle_replacement",
    ),
    AdversarialCaseSpec(
        "loop_executor_architecture_replacement",
        "authority",
        _AUDIT,
        "assembly_audit_fixture_loop_architecture",
    ),
    AdversarialCaseSpec(
        "loop_executor_optimizer_replacement",
        "authority",
        _AUDIT,
        "assembly_audit_fixture_loop_optimizer",
    ),
    AdversarialCaseSpec(
        "loop_executor_objective_replacement",
        "authority",
        _AUDIT,
        "assembly_audit_fixture_loop_objective",
    ),
    AdversarialCaseSpec(
        "validation_manual_happy_path_detected",
        "authority",
        _AUDIT,
        "assembly_audit_fixture_manual_lifecycle",
    ),
    AdversarialCaseSpec(
        "production_imports_validation_assembly",
        "authority",
        _AUDIT,
        "assembly_audit_fixture_production_validation_import",
    ),
    AdversarialCaseSpec(
        "competing_production_assembler_detected",
        "authority",
        _AUDIT,
        "assembly_audit_fixture_second_public_assembler",
    ),
)

assert tuple(item.case_id for item in ADVERSARIAL_CASE_SPECS) == ADVERSARIAL_CASE_IDS
