"""Focused, JAX-free closure audit for the post-P3.12 foundation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_student.validation.p3_12b_hf_descriptor_authority import (
    implementation_audit as p312b_audit,
)
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
# Every source package other than the proof-owned validation namespace is
# production or legacy support code.  This deliberately avoids a selective
# import-graph traversal, which could hide a new dependency in an unexercised
# product module.
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
    "arbitrary_architecture_compatibility_not_proven",
    "remote_ci_success_not_claimed",
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


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return tuple(sorted(names))


def _source_root(root: Path) -> Path:
    return root / "src" / "radjax_student"


def _production_paths(root: Path) -> tuple[Path, ...]:
    source = _source_root(root)
    return tuple(
        path
        for path in sorted(source.rglob("*.py"))
        if PROOF_OWNED_NAMESPACE not in path.parts
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
            "recorded_gates_read_only": self.status == "pass",
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
        for name in _imports(path)
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
        imports = _imports(path)
        if relative not in HISTORICAL_PROOF_EXCEPTIONS:
            validation_imports += sum(
                name.startswith("radjax_student.validation") for name in imports
            )
        if _has_proof_shape(path) and relative not in HISTORICAL_PROOF_EXCEPTIONS:
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
        imports = _imports(path)
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

    hf_shape = _hf_shape_current(repository)
    if not hf_shape:
        blockers.append("hf_shape_gate_failed")
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
        hf_shape,
        p312b_current,
        phase4_locked,
    )


def _p312b_recorded_evidence_current(root: Path) -> bool:
    try:
        payload = p312b_models.validate_receipt(
            json.loads(
                (root / "docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
            )
        )
        runner = (
            root
            / "src/radjax_student/validation/p3_12b_hf_descriptor_authority"
            / "runner_jax.py"
        )
        audit = p312b_audit.audit_gate_source(runner, repository_root=root)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        audit.status == "pass"
        and payload["implementation_audit_digest"] == audit.implementation_audit_digest
    )


def _hf_shape_current(root: Path) -> bool:
    required = {
        "architecture/models.py": (
            "ArchitectureInitResult",
            "hf_descriptor",
            "preservation_reference",
        ),
        "learning/assembly.py": (
            "initialized.hf_descriptor",
            "initialized.hf_reference",
        ),
        "checkpoints/v3.py": ("hf_descriptor", "preservation_reference"),
        "learning/run_report.py": ("RunHFSummary", "validate_hf_descriptor_match"),
    }
    source = _source_root(root)
    return all(
        all(token in (source / path).read_text(encoding="utf-8") for token in tokens)
        for path, tokens in required.items()
    )


def _phase4_policy_locked(root: Path) -> bool:
    roadmap = (root / "docs/RADJAX_DEVELOPMENT_ROADMAP.md").read_text(encoding="utf-8")
    return all(
        token in roadmap
        for token in (
            "Architecture Plugin Ingestion and First Real Architecture",
            "P4.1 — Architecture Ingestion Contract Freeze",
            "runtime RWKV mode",
            "Phase 4 architecture-plugin ingestion locally accepted",
        )
    )


def audit_source_fixture(source: str, *, relative_path: str) -> tuple[str, ...]:
    """Apply the closure's structural policy to one literal source fixture."""
    tree = ast.parse(source, filename=relative_path)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
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
    return tuple(sorted(set(blockers)))


def _bytes(report: FoundationAuditReport) -> bytes:
    return (
        json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check-recorded", action="store_true")
    args = parser.parse_args(argv)
    report = build_foundation_audit(Path.cwd())
    generated = _bytes(report)
    if args.write:
        REPORT_PATH.write_bytes(generated)
        return 0 if report.status == "pass" else 1
    return 0 if report.status == "pass" and REPORT_PATH.read_bytes() == generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
