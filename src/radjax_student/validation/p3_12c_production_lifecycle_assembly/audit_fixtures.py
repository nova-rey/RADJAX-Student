"""Focused JAX-free bad-source fixtures for the P3.12C authority audit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceAuditFixture:
    fixture_id: str
    source: str
    expected_blocker: str
    path: str = "src/radjax_student/learning/assembly.py"


REQUIRED_BAD_SOURCE_FIXTURES: tuple[SourceAuditFixture, ...] = (
    SourceAuditFixture(
        "second_public_assembler",
        "def assemble_jax_learning_lifecycle(): pass\ndef assemble_other(): pass\n",
        "assembly_audit_fixture_second_public_assembler",
    ),
    SourceAuditFixture(
        "assembler_under_validation",
        "def assemble_jax_learning_lifecycle(): pass\n",
        "assembly_audit_fixture_assembler_under_validation",
        "src/radjax_student/validation/p3_12c/assembly.py",
    ),
    SourceAuditFixture(
        "production_assembler_importing_validation",
        "from radjax_student.validation import gate\n",
        "assembly_audit_fixture_production_validation_import",
    ),
    SourceAuditFixture(
        "validation_manual_lifecycle",
        "JaxLearningLifecycle()\n",
        "assembly_audit_fixture_manual_lifecycle",
    ),
    SourceAuditFixture(
        "validation_manual_loop_executor",
        "JaxLoopExecutor()\n",
        "assembly_audit_fixture_manual_executor",
    ),
    SourceAuditFixture(
        "request_architecture_plugin",
        "class JaxLearningAssemblyRequest:\n    plugin: JaxArchitecturePlugin\n",
        "assembly_audit_fixture_request_architecture_plugin",
    ),
    SourceAuditFixture(
        "request_objective_plugin",
        "class JaxLearningAssemblyRequest:\n    plugin: ObjectivePlugin\n",
        "assembly_audit_fixture_request_objective_plugin",
    ),
    SourceAuditFixture(
        "request_optimizer_backend",
        "class JaxLearningAssemblyRequest:\n    backend: JaxOptimizerBackend\n",
        "assembly_audit_fixture_request_optimizer_backend",
    ),
    SourceAuditFixture(
        "request_runtime_backend",
        "class JaxLearningAssemblyRequest:\n    backend: ExecutionBackend\n",
        "assembly_audit_fixture_request_runtime_backend",
    ),
    SourceAuditFixture(
        "request_raw_loss_callable",
        "class JaxLearningAssemblyRequest:\n    loss: Callable\n",
        "assembly_audit_fixture_request_raw_loss_callable",
    ),
    SourceAuditFixture(
        "request_caller_hf_descriptor",
        "class JaxLearningAssemblyRequest:\n    hf: HFCompatibilityDescriptor\n",
        "assembly_audit_fixture_request_hf_descriptor",
    ),
    SourceAuditFixture(
        "assembler_fabricating_hf_descriptor",
        "def assemble_jax_learning_lifecycle():\n"
        "    return HFCompatibilityDescriptor()\n",
        "assembly_audit_fixture_hf_fabrication",
    ),
    SourceAuditFixture(
        "assembler_parameter_leaf_inspection",
        "def assemble_jax_learning_lifecycle():\n    return parameters['leaf']\n",
        "assembly_audit_fixture_parameter_leaves",
    ),
    SourceAuditFixture(
        "assembler_optimizer_leaf_inspection",
        "def assemble_jax_learning_lifecycle():\n    return optimizer_state.arrays\n",
        "assembly_audit_fixture_optimizer_leaves",
    ),
    SourceAuditFixture(
        "assembler_raw_jax_device",
        "def assemble_jax_learning_lifecycle():\n    return jax.devices()\n",
        "assembly_audit_fixture_raw_device",
    ),
    SourceAuditFixture(
        "competing_production_lifecycle_factory",
        "def build_other():\n    JaxLearningLifecycle()\n    JaxLoopExecutor()\n",
        "assembly_audit_fixture_competing_factory",
        "src/radjax_student/steps/other_factory.py",
    ),
    SourceAuditFixture(
        "caller_supplied_passed",
        "def build(passed=True):\n    return passed\n",
        "assembly_audit_fixture_caller_passed",
    ),
    SourceAuditFixture(
        "observer_expected_code",
        "def observe(invocation, expected_code):\n    return invocation\n",
        "assembly_audit_fixture_observer_expected_code",
    ),
    SourceAuditFixture(
        "permissive_expected_matcher",
        "def _matches_expected(observed, expected):\n    return observed == expected\n",
        "assembly_audit_fixture_permissive_matcher",
    ),
    SourceAuditFixture(
        "prefix_family_matcher",
        "def classify(value):\n    return value.startswith('assembly_')\n",
        "assembly_audit_fixture_prefix_matcher",
    ),
    SourceAuditFixture(
        "missing_exact_positive_inventory",
        "POSITIVE_CASE_IDS = ('wrong',)\n",
        "assembly_audit_fixture_positive_inventory",
    ),
    SourceAuditFixture(
        "missing_exact_adversarial_inventory",
        "ADVERSARIAL_CASE_IDS = ('wrong',)\n",
        "assembly_audit_fixture_adversarial_inventory",
    ),
    SourceAuditFixture(
        "receipt_success_flag",
        "def build_receipt(proof, success=False):\n    return proof\n",
        "assembly_audit_fixture_receipt_success_flag",
    ),
    SourceAuditFixture(
        "validation_case_id_in_production_assembler",
        "def assemble_jax_learning_lifecycle(case_id):\n    return case_id\n",
        "assembly_audit_fixture_validation_case_id",
    ),
)

assert len(REQUIRED_BAD_SOURCE_FIXTURES) == 24


__all__ = ["REQUIRED_BAD_SOURCE_FIXTURES", "SourceAuditFixture"]
