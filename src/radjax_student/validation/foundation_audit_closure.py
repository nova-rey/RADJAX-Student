"""Focused, JAX-free closure audit for the post-P3.12 foundation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_student.validation.p3_12b_hf_descriptor_authority import (
    models as p312b_models,
)

SCHEMA_VERSION = "radjax.foundation_audit_closure_report.v1"
REPORT_PATH = Path("docs/FOUNDATION_AUDIT_CLOSURE_REPORT.json")

# This is intentionally a reviewed literal contract, not import-graph discovery.
CANONICAL_TRAINING_PATHS = (
    "architecture/models.py",
    "architecture/protocols.py",
    "architecture/registry.py",
    "objectives/builtin.py",
    "objectives/jax.py",
    "objectives/registry.py",
    "optimizers/jax.py",
    "optimizers/registry.py",
    "runtime/callables.py",
    "runtime/execution.py",
    "runtime/keys.py",
    "runtime/lifecycle.py",
    "learning/assembly.py",
    "learning/composition.py",
    "learning/jax_batch.py",
    "learning/jax_core.py",
    "learning/jax_execution.py",
    "steps/jax_loop.py",
    "steps/jax_step.py",
)
# These owners are a reviewed production boundary, not an import-graph
# approximation.  A proof-shaped module beneath any of them is a policy
# violation unless it is one of the frozen paths below.
PRODUCTION_OWNER_ROOTS = frozenset(
    {
        "architecture",
        "checkpoints",
        "contracts",
        "hf",
        "learning",
        "objectives",
        "optimizers",
        "reports",
        "runtime",
        "steps",
    }
)
PROOF_OWNED_NAMESPACE = "validation"
HISTORICAL_PROOF_EXCEPTIONS = (
    "learning/observability_acceptance.py",
    "learning/p3_10_acceptance.py",
    "learning/p3_5_acceptance.py",
    "learning/synthetic_smoke.py",
)
CLAIMS_NOT_MADE = (
    "historical_acceptance_modules_not_fully_relocated",
    "validation_namespace_not_fully_split",
    "arbitrary_architecture_ingestion_not_yet_proven",
    "rwkv_not_implemented",
    "phase4_not_started",
    "full_hf_export_not_implemented",
    "save_pretrained_not_implemented",
    "from_pretrained_not_implemented",
    "teacher_inference_not_implemented",
    "tome_training_not_started",
    "distributed_training_not_proven",
    "multi_device_training_not_proven",
    "tpu_training_not_proven",
    "pallas_optimization_not_started",
    "production_cli_not_implemented",
    "model_quality_not_measured",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _normalized_from_imports(
    node: ast.ImportFrom, *, relative_path: str
) -> tuple[str, ...]:
    """Resolve a relative import from a source-relative module path."""
    if not node.level:
        return (node.module,) if node.module else ()
    package = ("radjax_student", *Path(relative_path).parent.parts)
    parent_count = node.level - 1
    if parent_count >= len(package):
        return ()
    base = ".".join(package[: len(package) - parent_count])
    if node.module:
        return (f"{base}.{node.module}",)
    return tuple(f"{base}.{item.name}" for item in node.names)


def _imports_from_tree(tree: ast.Module, *, relative_path: str) -> tuple[str, ...]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(_normalized_from_imports(node, relative_path=relative_path))
    return tuple(sorted(names))


def _imports(path: Path, *, relative_path: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _imports_from_tree(tree, relative_path=relative_path)


def _source_root(root: Path) -> Path:
    return root / "src" / "radjax_student"


def _production_paths(root: Path) -> tuple[Path, ...]:
    source = _source_root(root)
    return tuple(
        path
        for path in sorted(source.rglob("*.py"))
        if path.relative_to(source).parts[0] in PRODUCTION_OWNER_ROOTS
    )


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(_source_root(root)))


def _has_proof_shape(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            node.name.startswith(("run_p3_", "execute_p3_"))
            or node.name.endswith("_acceptance")
            or node.name.endswith("_proof")
        )
        for node in tree.body
    ) or any(
        isinstance(node, ast.ClassDef)
        and ("Acceptance" in node.name or node.name.endswith("Proof"))
        for node in tree.body
    )


def _is_new_proof_path(relative: str) -> bool:
    """Reject proof/acceptance filenames under production owners exactly."""
    if relative in HISTORICAL_PROOF_EXCEPTIONS:
        return False
    filename = Path(relative).stem
    return any(token in filename for token in ("acceptance", "proof", "audit"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def _call_keywords(function: ast.FunctionDef, callee: str) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == callee:
            keys.update(keyword.arg for keyword in node.keywords if keyword.arg)
    return keys


def _call_keyword_attribute(
    function: ast.FunctionDef,
    callee: str,
    keyword_name: str,
    value: str,
    attribute: str,
) -> bool:
    return any(
        _call_name(node.func) == callee
        and any(
            keyword.arg == keyword_name
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == value
            and keyword.value.attr == attribute
            for keyword in node.keywords
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    )


def _call_arguments_are(node: ast.Call, callee: str, *arguments: str) -> bool:
    return (
        _call_name(node.func) == callee
        and tuple(_call_name(argument) for argument in node.args) == arguments
    )


def _has_operational_report_validation(function: ast.FunctionDef) -> bool:
    """Require the fixed direct try-path; dead calls do not prove validation."""
    for statement in function.body:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return False
        if isinstance(statement, ast.Try):
            for item in statement.body:
                if isinstance(item, ast.Raise) or _may_return(item):
                    return False
                if (
                    isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Call)
                    and _call_arguments_are(
                        item.value,
                        "validate_hf_descriptor_match",
                        "executed_descriptor",
                        "summary.descriptor",
                    )
                ):
                    return True
            return False
    return False


def _may_return(statement: ast.stmt) -> bool:
    """Conservatively detect an outer-function return before the matcher."""
    if isinstance(statement, ast.Return):
        return True
    if isinstance(statement, ast.If):
        if isinstance(statement.test, ast.Constant) and isinstance(
            statement.test.value, bool
        ):
            branches = statement.body if statement.test.value else statement.orelse
        else:
            branches = (*statement.body, *statement.orelse)
        return any(_may_return(item) for item in branches)
    if isinstance(
        statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)
    ):
        return any(_may_return(item) for item in statement.body)
    if isinstance(statement, ast.Try):
        return any(
            _may_return(item)
            for item in (*statement.body, *statement.orelse, *statement.finalbody)
        ) or any(
            _may_return(item) for handler in statement.handlers for item in handler.body
        )
    return False


def _handler_swallows_checkpoint_mismatch(handler: ast.ExceptHandler) -> bool:
    def catches_checkpoint_error(caught: ast.expr | None) -> bool:
        if caught is None:
            return True
        if isinstance(caught, ast.Name):
            return caught.id in {
                "CheckpointValidationError",
                "Exception",
                "BaseException",
            }
        if isinstance(caught, ast.Tuple):
            return any(catches_checkpoint_error(item) for item in caught.elts)
        return False

    return catches_checkpoint_error(handler.type) and not any(
        isinstance(statement, ast.Raise) for statement in handler.body
    )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _contains_attribute(tree: ast.AST, value: str, attribute: str) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == value
        and node.attr == attribute
        for node in ast.walk(tree)
    )


def _authority_blockers(sources: dict[str, str]) -> tuple[str, ...]:
    """Prove the fixed HF descriptor route using source structure, not tokens."""
    blockers: list[str] = []

    def tree(path: str) -> ast.Module | None:
        source = sources.get(path)
        if source is None:
            blockers.append(f"hf_authority_source_missing:{path}")
            return None
        return ast.parse(source, filename=path)

    models = tree("architecture/models.py")
    if models is not None:
        result = next(
            (
                node
                for node in models.body
                if isinstance(node, ast.ClassDef)
                and node.name == "ArchitectureInitResult"
            ),
            None,
        )
        if result is None or not (
            _contains_attribute(result, "self", "hf_descriptor")
            and _contains_attribute(result, "self", "hf_reference")
            and any(
                isinstance(node, ast.Call)
                and _call_name(node.func) == "self.hf_descriptor.preservation_reference"
                for node in ast.walk(result)
            )
        ):
            blockers.append("hf_architecture_descriptor_reference_unproven")

    assembly = tree("learning/assembly.py")
    if assembly is not None:
        assemble = _function(assembly, "assemble_jax_learning_lifecycle")
        if assemble is None or not (
            _contains_attribute(assemble, "initialized", "hf_descriptor")
            and _contains_attribute(assemble, "initialized", "hf_reference")
            and _call_keyword_attribute(
                assemble,
                "JaxLearningLifecycle",
                "hf_descriptor",
                "initialized",
                "hf_descriptor",
            )
            and _call_keyword_attribute(
                assemble,
                "JaxLearningLifecycle",
                "hf_reference",
                "initialized",
                "hf_reference",
            )
        ):
            blockers.append("hf_assembly_descriptor_substitution")

    lifecycle = tree("steps/jax_loop.py")
    if lifecycle is not None:
        restore = _function(lifecycle, "restore_from_checkpoint")
        if restore is None or not (
            _call_keyword_attribute(
                restore,
                "load_learning_checkpoint_v3",
                "expected_hf_reference",
                "self",
                "hf_reference",
            )
            and _call_keyword_attribute(
                restore,
                "load_learning_checkpoint_v3",
                "expected_hf_descriptor",
                "self",
                "hf_descriptor",
            )
        ):
            blockers.append("hf_checkpoint_lifecycle_bypass")

    checkpoint = tree("checkpoints/v3.py")
    if checkpoint is not None:
        restore = _function(checkpoint, "load_learning_checkpoint_v3")
        has_required_guard = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Is)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "expected_hf_descriptor"
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
            for node in ast.walk(restore or checkpoint)
        )
        mismatch_rejects = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.NotEq)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "hf_descriptor"
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Name)
            and node.test.comparators[0].id == "expected_hf_descriptor"
            and any(isinstance(statement, ast.Raise) for statement in node.body)
            for node in ast.walk(restore or checkpoint)
        )
        mismatch_is_swallowed = any(
            any(
                _handler_swallows_checkpoint_mismatch(handler)
                for handler in node.handlers
            )
            and any(
                isinstance(descendant, ast.If)
                and isinstance(descendant.test, ast.Compare)
                and len(descendant.test.ops) == 1
                and isinstance(descendant.test.ops[0], ast.NotEq)
                and isinstance(descendant.test.left, ast.Name)
                and descendant.test.left.id == "hf_descriptor"
                and len(descendant.test.comparators) == 1
                and isinstance(descendant.test.comparators[0], ast.Name)
                and descendant.test.comparators[0].id == "expected_hf_descriptor"
                for descendant in ast.walk(node)
            )
            for node in ast.walk(restore or checkpoint)
            if isinstance(node, ast.Try)
        )
        if (
            restore is None
            or not has_required_guard
            or not mismatch_rejects
            or mismatch_is_swallowed
        ):
            blockers.append("hf_checkpoint_descriptor_validation_bypassed")

    replay = tree("validation/p3_11_9_replay/runner_jax.py")
    if replay is not None:
        identity = _function(replay, "_identity")
        if identity is None or not (
            _contains_attribute(identity, "lifecycle", "hf_descriptor")
            and _contains_attribute(identity, "lifecycle", "hf_reference")
        ):
            blockers.append("hf_replay_non_lifecycle_descriptor")

    report = tree("learning/run_report.py")
    if report is not None:
        validate = _function(report, "validate_run_hf_summary")
        if validate is None or not _has_operational_report_validation(validate):
            blockers.append("hf_report_descriptor_validation_bypassed")
    return tuple(sorted(set(blockers)))


@dataclass(frozen=True)
class FoundationAuditReport:
    blockers: tuple[str, ...]
    runtime_steps_import_count: int
    runtime_learning_import_count: int
    production_validation_import_count: int
    canonical_numpy_loss_import_count: int
    test_support_hermetic: bool
    jax_purity_gate: bool
    hf_shape_gate: bool
    p312b_recorded_evidence_current: bool
    phase4_ingestion_policy_locked: bool

    @property
    def composition_root_corrected(self) -> bool:
        return (
            self.runtime_steps_import_count == 0
            and self.runtime_learning_import_count == 0
        )

    @property
    def status(self) -> str:
        return "pass" if not self.blockers else "fail"

    @property
    def evidence_digest(self) -> str:
        return _digest(
            {
                "status": self.status,
                "composition_root_corrected": self.composition_root_corrected,
                "runtime_steps_import_count": self.runtime_steps_import_count,
                "runtime_learning_import_count": self.runtime_learning_import_count,
                "production_validation_import_count": (
                    self.production_validation_import_count
                ),
                "canonical_numpy_loss_import_count": (
                    self.canonical_numpy_loss_import_count
                ),
                "test_support_hermetic": self.test_support_hermetic,
                "jax_purity_gate": self.jax_purity_gate,
                "hf_shape_gate": self.hf_shape_gate,
                "p312b_recorded_evidence_current": self.p312b_recorded_evidence_current,
                "phase4_ingestion_policy_locked": self.phase4_ingestion_policy_locked,
                "claims_not_made": list(CLAIMS_NOT_MADE),
                "blockers": list(self.blockers),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "composition_root_corrected": self.composition_root_corrected,
            "runtime_steps_import_count": self.runtime_steps_import_count,
            "runtime_learning_import_count": self.runtime_learning_import_count,
            "production_validation_import_count": (
                self.production_validation_import_count
            ),
            "canonical_numpy_loss_import_count": self.canonical_numpy_loss_import_count,
            "test_support_hermetic": self.test_support_hermetic,
            "jax_purity_gate": self.jax_purity_gate,
            "hf_shape_gate": self.hf_shape_gate,
            "p312b_recorded_evidence_current": self.p312b_recorded_evidence_current,
            "phase4_ingestion_policy_locked": self.phase4_ingestion_policy_locked,
            "claims_not_made": list(CLAIMS_NOT_MADE),
            "blockers": list(self.blockers),
            "evidence_digest": self.evidence_digest,
        }


def build_foundation_audit(root: Path | None = None) -> FoundationAuditReport:
    """Measure the frozen closure boundaries without importing JAX surfaces."""
    repository = Path.cwd() if root is None else root
    source = _source_root(repository)
    blockers: list[str] = []
    runtime_imports = tuple(
        name
        for path in sorted((source / "runtime").rglob("*.py"))
        for name in _imports(path, relative_path=_relative(repository, path))
    )
    runtime_steps = sum(
        name.startswith("radjax_student.steps") for name in runtime_imports
    )
    runtime_learning = sum(
        name.startswith("radjax_student.learning") for name in runtime_imports
    )
    if runtime_steps:
        blockers.append("runtime_steps_import")
    if runtime_learning:
        blockers.append("runtime_learning_import")

    validation_imports = 0
    for path in _production_paths(repository):
        relative = _relative(repository, path)
        imports = _imports(path, relative_path=relative)
        if relative not in HISTORICAL_PROOF_EXCEPTIONS:
            validation_imports += sum(
                name.startswith("radjax_student.validation") for name in imports
            )
        if (
            _is_new_proof_path(relative)
            or _has_proof_shape(path)
            and (relative not in HISTORICAL_PROOF_EXCEPTIONS)
        ):
            blockers.append(f"new_production_proof_module:{relative}")
    if validation_imports:
        blockers.append("production_validation_import")

    canonical_loss_imports = 0
    purity_failures: list[str] = []
    forbidden = (
        "numpy",
        "torch",
        "tensorflow",
        "tensorflow_probability",
        "transformers",
    )
    for relative in CANONICAL_TRAINING_PATHS:
        path = source / relative
        imports = _imports(path, relative_path=relative)
        canonical_loss_imports += sum(
            name.startswith("radjax_student.legacy.losses") for name in imports
        )
        if any(name.split(".", 1)[0] in forbidden for name in imports):
            purity_failures.append(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Attribute) and node.attr in {"numpy", "tolist"}
            for node in ast.walk(tree)
        ):
            purity_failures.append(relative)
    if canonical_loss_imports:
        blockers.append("canonical_numpy_loss_import")
    if purity_failures:
        blockers.append(
            "canonical_jax_purity:" + ",".join(sorted(set(purity_failures)))
        )

    support = repository / "tests" / "support" / "linear_objective.py"
    test_support_hermetic = (
        support.is_file() and (repository / "tests" / "__init__.py").is_file()
    )
    if not test_support_hermetic:
        blockers.append("test_support_not_hermetic")

    hf_authority_blockers = _hf_shape_blockers(repository)
    if hf_authority_blockers:
        blockers.extend(hf_authority_blockers)
    p312b_current = _p312b_recorded_evidence_current(repository)
    if not p312b_current:
        blockers.append("p312b_recorded_evidence_stale")
    phase4_locked = _phase4_policy_locked(repository)
    if not phase4_locked:
        blockers.append("phase4_ingestion_policy_missing")
    return FoundationAuditReport(
        tuple(sorted(set(blockers))),
        runtime_steps,
        runtime_learning,
        validation_imports,
        canonical_loss_imports,
        test_support_hermetic,
        not purity_failures and canonical_loss_imports == 0,
        not hf_authority_blockers,
        p312b_current,
        phase4_locked,
    )


def _p312b_recorded_evidence_current(root: Path) -> bool:
    try:
        payload = json.loads(
            (root / "docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
        )
        from radjax_student.validation.p3_12b_hf_descriptor_authority import (
            runner_jax,
        )

        with tempfile.TemporaryDirectory(prefix="radjax-foundation-p312b-") as temp:
            proof = runner_jax.execute_hf_descriptor_authority_proof(Path(temp))
        p312b_models.validate_receipt(payload, proof=proof)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


HF_AUTHORITY_PATHS = (
    "architecture/models.py",
    "learning/assembly.py",
    "steps/jax_loop.py",
    "checkpoints/v3.py",
    "validation/p3_11_9_replay/runner_jax.py",
    "learning/run_report.py",
)


def _hf_shape_blockers(root: Path) -> tuple[str, ...]:
    source = _source_root(root)
    try:
        sources = {
            relative: (source / relative).read_text(encoding="utf-8")
            for relative in HF_AUTHORITY_PATHS
        }
    except OSError:
        return ("hf_authority_source_unreadable",)
    return _authority_blockers(sources)


def _phase4_policy_locked(root: Path) -> bool:
    roadmap = (root / "docs/RADJAX_DEVELOPMENT_ROADMAP.md").read_text(encoding="utf-8")
    return all(
        token in roadmap
        for token in (
            "Architecture Plugin Ingestion and First Real Architecture",
            "P4.1 — Architecture Ingestion Contract Freeze",
            "runtime RWKV mode",
            "Phase 4 remains unstarted",
        )
    )


def audit_source_fixture(source: str, *, relative_path: str) -> tuple[str, ...]:
    """Apply the closure's structural policy to one literal source fixture."""
    tree = ast.parse(source, filename=relative_path)
    imports = _imports_from_tree(tree, relative_path=relative_path)
    blockers: list[str] = []
    if relative_path.startswith("runtime/"):
        if any(name.startswith("radjax_student.steps") for name in imports):
            blockers.append("runtime_steps_import")
        if any(name.startswith("radjax_student.learning") for name in imports):
            blockers.append("runtime_learning_import")
    if relative_path.split("/", 1)[0] != PROOF_OWNED_NAMESPACE and any(
        name.startswith("radjax_student.validation") for name in imports
    ):
        blockers.append("production_validation_import")
    if relative_path in CANONICAL_TRAINING_PATHS and any(
        name.split(".", 1)[0]
        in {"numpy", "torch", "tensorflow", "tensorflow_probability", "transformers"}
        for name in imports
    ):
        blockers.append("canonical_jax_purity")
    if relative_path in CANONICAL_TRAINING_PATHS and any(
        name.startswith("radjax_student.legacy.losses") for name in imports
    ):
        blockers.append("canonical_numpy_loss_import")
    if relative_path.split("/", 1)[0] in PRODUCTION_OWNER_ROOTS and (
        _is_new_proof_path(relative_path)
        or _has_proof_shape_from_tree(tree)
        and relative_path not in HISTORICAL_PROOF_EXCEPTIONS
    ):
        blockers.append(f"new_production_proof_module:{relative_path}")
    return tuple(sorted(set(blockers)))


def _has_proof_shape_from_tree(tree: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            node.name.startswith(("run_p3_", "execute_p3_"))
            or node.name.endswith("_acceptance")
            or node.name.endswith("_proof")
        )
        for node in tree.body
    ) or any(
        isinstance(node, ast.ClassDef)
        and ("Acceptance" in node.name or node.name.endswith("Proof"))
        for node in tree.body
    )


def audit_hf_authority_fixture(sources: dict[str, str]) -> tuple[str, ...]:
    """Expose fixed-path HF provenance checks for independent breakage tests."""
    return _authority_blockers(sources)


def _bytes(report: FoundationAuditReport) -> bytes:
    return (
        json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check-recorded", action="store_true")
    parser.add_argument("--recorded", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    report = build_foundation_audit(Path.cwd())
    generated = _bytes(report)
    if args.write:
        args.recorded.write_bytes(generated)
        return 0 if report.status == "pass" else 1
    return (
        0 if report.status == "pass" and args.recorded.read_bytes() == generated else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
