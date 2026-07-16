"""Literal Section J isolated dependency-audit experiments."""

from __future__ import annotations

import shutil
from pathlib import Path

from radjax_student.validation.architecture_audit import (
    build_architecture_audit,
    validate_dependency_fixture,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_directory_experiment,
    public_boundary,
)


def _clean_tree(root: Path) -> Path:
    package = root / "src" / "radjax_student"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    return root


def _clone_tree(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _write_module(root: Path, relative: str, source: str) -> None:
    path = root / "src" / "radjax_student" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


@public_boundary("dependency_audit_validation")
def _audit(root: Path) -> None:
    validate_dependency_fixture(build_architecture_audit(root))


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
        public_input_kind="isolated_dependency_source_tree",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=_audit,
        baseline_callable=_audit,
    )


def experiment_j_maintained_dependency_audit_exact_accepted_edges(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "clean-baseline")
    mutated = _clone_tree(baseline, context.temporary_root / "clean-mutated")
    _write_module(mutated, "contracts/marker.py", "VALUE = 'clean'\n")
    return _record(
        context, baseline, mutated, "contracts/marker.py", "add_allowed_contract_module"
    )


def experiment_j_architecture_imports_validation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "architecture-validation-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "architecture-validation-mutated"
    )
    _write_module(
        mutated,
        "architecture/plugin.py",
        "from radjax_student.validation import probe\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "architecture/plugin.py",
        "insert_architecture_validation_import",
    )


def experiment_j_optimizer_imports_validation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "optimizer-validation-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "optimizer-validation-mutated"
    )
    _write_module(
        mutated,
        "optimizers/backend.py",
        "from radjax_student.validation import probe\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "optimizers/backend.py",
        "insert_optimizer_validation_import",
    )


def experiment_j_runtime_imports_validation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "runtime-validation-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "runtime-validation-mutated"
    )
    _write_module(
        mutated, "runtime/backend.py", "from radjax_student.validation import probe\n"
    )
    return _record(
        context,
        baseline,
        mutated,
        "runtime/backend.py",
        "insert_runtime_validation_import",
    )


def experiment_j_checkpoint_imports_validation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "checkpoint-validation-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "checkpoint-validation-mutated"
    )
    _write_module(
        mutated, "checkpoints/store.py", "from radjax_student.validation import probe\n"
    )
    return _record(
        context,
        baseline,
        mutated,
        "checkpoints/store.py",
        "insert_checkpoint_validation_import",
    )


def experiment_j_generic_learning_imports_jax_execution(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "learning-jax-baseline")
    mutated = _clone_tree(baseline, context.temporary_root / "learning-jax-mutated")
    _write_module(
        mutated,
        "learning/loop.py",
        "from radjax_student.steps.jax_step import execute_jax_learning_step\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "learning/loop.py",
        "insert_generic_learning_jax_step_import",
    )


def experiment_j_objective_imports_concrete_architecture_parameters(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "objective-parameters-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "objective-parameters-mutated"
    )
    _write_module(
        mutated,
        "learning/objective.py",
        "def run(objective, parameters, batch):\n"
        "    return objective.evaluate(parameters, batch)\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "learning/objective.py",
        "pass_parameters_to_objective_evaluate",
    )


def experiment_j_architecture_imports_optimizer_implementation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "architecture-optimizer-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "architecture-optimizer-mutated"
    )
    _write_module(
        mutated,
        "architecture/plugin.py",
        "from radjax_student.optimizers.sgd import SgdOptimizer\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "architecture/plugin.py",
        "insert_architecture_optimizer_import",
    )


def experiment_j_optimizer_imports_architecture_implementation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "optimizer-architecture-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "optimizer-architecture-mutated"
    )
    _write_module(
        mutated,
        "optimizers/backend.py",
        "from radjax_student.architecture.testing import FakeArchitecturePlugin\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "optimizers/backend.py",
        "insert_optimizer_architecture_import",
    )


def experiment_j_runtime_imports_architecture_or_optimizer_implementation(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "runtime-architecture-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "runtime-architecture-mutated"
    )
    _write_module(
        mutated,
        "runtime/backend.py",
        "from radjax_student.architecture.testing import FakeArchitecturePlugin\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "runtime/backend.py",
        "insert_runtime_architecture_import",
    )


def experiment_j_learning_imports_legacy_jax_helper(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "learning-legacy-baseline")
    mutated = _clone_tree(baseline, context.temporary_root / "learning-legacy-mutated")
    _write_module(
        mutated, "learning/loop.py", "from radjax_student.legacy.scalar import update\n"
    )
    return _record(
        context,
        baseline,
        mutated,
        "learning/loop.py",
        "insert_learning_legacy_helper_import",
    )


def experiment_j_production_imports_validation_stateful_architecture(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "production-stateful-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "production-stateful-mutated"
    )
    _write_module(
        mutated,
        "architecture/production.py",
        "from radjax_student.validation.p3_11_9_replay.runner_jax "
        "import StatefulLinearJaxArchitecture\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "architecture/production.py",
        "import_validation_stateful_architecture",
    )


def experiment_j_production_imports_replay_runner(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "production-replay-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "production-replay-mutated"
    )
    _write_module(
        mutated,
        "learning/production.py",
        "from radjax_student.validation.p3_11_9_replay import runner_jax\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "learning/production.py",
        "import_validation_replay_runner",
    )


def experiment_j_validation_imported_by_public_package_initialization(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "package-validation-baseline")
    mutated = _clone_tree(
        baseline, context.temporary_root / "package-validation-mutated"
    )
    _write_module(
        mutated, "__init__.py", "from radjax_student.validation import gate\n"
    )
    return _record(
        context,
        baseline,
        mutated,
        "__init__.py",
        "insert_public_package_validation_import",
    )


def experiment_j_students_registry_reintroduced(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "students-baseline")
    mutated = _clone_tree(baseline, context.temporary_root / "students-mutated")
    _write_module(
        mutated,
        "__init__.py",
        "from radjax_student.students import StudentBackendRegistry\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "__init__.py",
        "reintroduce_students_registry_import",
    )


def experiment_j_legacy_exported_current_production_namespace(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "legacy-export-baseline")
    mutated = _clone_tree(baseline, context.temporary_root / "legacy-export-mutated")
    _write_module(
        mutated,
        "architecture/__init__.py",
        "from radjax_student.legacy.scalar import update\n__all__ = ['update']\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "architecture/__init__.py",
        "export_legacy_helper_from_architecture_namespace",
    )


def experiment_j_duplicated_contract_class_alias(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "duplicate-contract-baseline")
    _write_module(baseline, "contracts/batch.py", "VALUE = 'canonical'\n")
    mutated = _clone_tree(
        baseline, context.temporary_root / "duplicate-contract-mutated"
    )
    _write_module(mutated, "contracts/batch.py", "class LearningBatch:\n    pass\n")
    return _record(
        context,
        baseline,
        mutated,
        "contracts/batch.py",
        "define_duplicate_learning_batch_class",
    )


def experiment_j_dependency_cycle_introduction(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _clean_tree(context.temporary_root / "cycle-baseline")
    _write_module(baseline, "architecture/plugin.py", "VALUE = 'clean'\n")
    mutated = _clone_tree(baseline, context.temporary_root / "cycle-mutated")
    _write_module(
        mutated,
        "architecture/plugin.py",
        "from radjax_student.optimizers.backend import VALUE\n",
    )
    _write_module(
        mutated,
        "optimizers/backend.py",
        "from radjax_student.architecture.plugin import VALUE\n",
    )
    return _record(
        context,
        baseline,
        mutated,
        "architecture/plugin.py -> optimizers/backend.py",
        "insert_architecture_optimizer_dependency_cycle",
    )


SECTION_IMPLEMENTATIONS = {
    "J.positive.maintained_dependency_audit_exact_accepted_edges": GateCaseImplementation(  # noqa: E501
        experiment_j_maintained_dependency_audit_exact_accepted_edges
    ),
    "J.reject.architecture_imports_validation": GateCaseImplementation(
        experiment_j_architecture_imports_validation
    ),
    "J.reject.optimizer_imports_validation": GateCaseImplementation(
        experiment_j_optimizer_imports_validation
    ),
    "J.reject.runtime_imports_validation": GateCaseImplementation(
        experiment_j_runtime_imports_validation
    ),
    "J.reject.checkpoint_imports_validation": GateCaseImplementation(
        experiment_j_checkpoint_imports_validation
    ),
    "J.reject.generic_learning_imports_jax_execution": GateCaseImplementation(
        experiment_j_generic_learning_imports_jax_execution
    ),
    "J.reject.objective_imports_concrete_architecture_parameters": GateCaseImplementation(  # noqa: E501
        experiment_j_objective_imports_concrete_architecture_parameters
    ),
    "J.reject.architecture_imports_optimizer_implementation": GateCaseImplementation(
        experiment_j_architecture_imports_optimizer_implementation
    ),
    "J.reject.optimizer_imports_architecture_implementation": GateCaseImplementation(
        experiment_j_optimizer_imports_architecture_implementation
    ),
    "J.reject.runtime_imports_architecture_or_optimizer_implementation": GateCaseImplementation(  # noqa: E501
        experiment_j_runtime_imports_architecture_or_optimizer_implementation
    ),
    "J.reject.learning_imports_legacy_jax_helper": GateCaseImplementation(
        experiment_j_learning_imports_legacy_jax_helper
    ),
    "J.reject.production_imports_validation_stateful_architecture": GateCaseImplementation(  # noqa: E501
        experiment_j_production_imports_validation_stateful_architecture
    ),
    "J.reject.production_imports_replay_runner": GateCaseImplementation(
        experiment_j_production_imports_replay_runner
    ),
    "J.reject.validation_imported_by_public_package_initialization": GateCaseImplementation(  # noqa: E501
        experiment_j_validation_imported_by_public_package_initialization
    ),
    "J.reject.students_registry_reintroduced": GateCaseImplementation(
        experiment_j_students_registry_reintroduced
    ),
    "J.reject.legacy_exported_current_production_namespace": GateCaseImplementation(
        experiment_j_legacy_exported_current_production_namespace
    ),
    "J.reject.duplicated_contract_class_alias": GateCaseImplementation(
        experiment_j_duplicated_contract_class_alias
    ),
    "J.reject.dependency_cycle_introduction": GateCaseImplementation(
        experiment_j_dependency_cycle_introduction
    ),
}
