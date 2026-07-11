"""P3.5.10 adversarial architecture-integrity acceptance receipt."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from scripts.audit_architecture import build_audit

SCHEMA = "radjax.p3_5_architecture_integrity_receipt.v1"
FLAGS = (
    "dependency_boundaries_valid",
    "architecture_objective_separation_valid",
    "jax_native_learning_contract_valid",
    "architecture_namespace_valid",
    "legacy_isolation_valid",
    "hf_preservation_contract_valid",
    "checkpoint_ownership_valid",
    "documentation_consistency_valid",
    "phase1_regression_valid",
    "phase2_regression_valid",
    "phase3_regression_valid",
    "import_purity_valid",
    "deterministic_replay_valid",
)


@dataclass(frozen=True)
class P35ArchitectureIntegrityReceipt:
    schema_version: str
    status: Literal["pass", "fail"]
    dependency_boundaries_valid: bool
    architecture_objective_separation_valid: bool
    jax_native_learning_contract_valid: bool
    architecture_namespace_valid: bool
    legacy_isolation_valid: bool
    hf_preservation_contract_valid: bool
    checkpoint_ownership_valid: bool
    documentation_consistency_valid: bool
    phase1_regression_valid: bool
    phase2_regression_valid: bool
    phase3_regression_valid: bool
    import_purity_valid: bool
    deterministic_replay_valid: bool
    blockers: tuple[str, ...] = ()
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA:
            raise ValueError("invalid P3.5 receipt schema")
        values = {name: getattr(self, name) for name in FLAGS}
        if any(type(value) is not bool for value in values.values()):
            raise TypeError("P3.5 receipt flags must be booleans")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, str) or not item for item in blockers):
            raise TypeError("P3.5 receipt blockers must be nonempty strings")
        expected_status = all(values.values()) and not blockers
        if (self.status == "pass") != expected_status:
            raise ValueError("P3.5 receipt status does not match flags")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            **{name: getattr(self, name) for name in FLAGS},
            "blockers": list(self.blockers),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _source_root(repo_root: Path) -> Path:
    return repo_root / "src" / "radjax_student"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def _run_jax_contract() -> tuple[bool, bool, str | None]:
    if importlib.util.find_spec("jax") is None:
        return False, False, "jax is unavailable in the acceptance environment"
    try:
        import jax
        import jax.numpy as jnp

        from radjax_student.architecture import ForwardResult
        from radjax_student.learning.jax_core import (
            JaxBatch,
            JaxObjectiveConfig,
            apply_scoped_gradient_update,
            build_jax_loss_fn,
            build_value_and_grad_fn,
            validate_finite_loss_and_gradients,
        )

        class LinearArchitecture:
            def apply_jax(
                self,
                parameters,
                architecture_state,
                batch,
                *,
                objective_scope,
                training,
                rng_key,
            ):
                del objective_scope, training, rng_key
                prediction = (
                    parameters["weight"] * batch.inputs + architecture_state["bias"]
                )
                return ForwardResult(
                    surface_values={"prediction": prediction},
                    updated_runtime_state=architecture_state,
                )

        class MeanSquaredError:
            def evaluate(self, surface, targets, weights, objective_config):
                del weights, objective_config
                return jnp.mean((surface - targets) ** 2), {}

        loss_fn = build_jax_loss_fn(LinearArchitecture(), MeanSquaredError())
        value_and_grad = build_value_and_grad_fn(loss_fn)
        args = (
            {"weight": jnp.asarray(1.0)},
            {"bias": jnp.asarray(1.0)},
            JaxBatch(jnp.asarray(2.0), jnp.asarray(5.0)),
            JaxObjectiveConfig("mse", surface_id="prediction"),
            jax.random.key(0),
        )
        first = value_and_grad(*args)
        second = value_and_grad(*args)
        validate_finite_loss_and_gradients(first[0][0], first[1])
        replay = bool(jnp.allclose(first[0][0], second[0][0])) and bool(
            jnp.allclose(first[1]["weight"], second[1]["weight"])
        )
        updated = apply_scoped_gradient_update(args[0], first[1], {"weight": True}, 0.1)
        return True, replay and bool(updated["weight"] != args[0]["weight"]), None
    except Exception as exc:  # The receipt records the failed contract seam.
        return False, False, f"JAX contract execution failed: {type(exc).__name__}"


def run_p3_5_architecture_integrity_acceptance(
    repo_root: Path | None = None,
) -> P35ArchitectureIntegrityReceipt:
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    source_root = _source_root(root)
    blockers: list[str] = []
    audit = build_audit(root)
    dependency_ok = audit["status"] == "pass" and not audit["blockers"]
    if not dependency_ok:
        blockers.append("dependency audit contains blockers")

    jax_source = source_root / "learning" / "jax_core.py"
    architecture_source = source_root / "architecture"
    architecture_ok = (
        "objective.evaluate(\n            surface"
        in jax_source.read_text(encoding="utf-8")
        and all(
            "radjax_student.students" not in path.read_text(encoding="utf-8")
            for path in architecture_source.rglob("*.py")
        )
    )
    if not architecture_ok:
        blockers.append("architecture/objective separation is not explicit")

    jax_ok, replay_ok, jax_blocker = _run_jax_contract()
    if jax_blocker:
        blockers.append(jax_blocker)

    root_source = (source_root / "__init__.py").read_text(encoding="utf-8")
    namespace_ok = (
        "ArchitectureRegistry"
        in (architecture_source / "__init__.py").read_text(encoding="utf-8")
        and "students" not in root_source
        and all(
            not any(
                name.startswith("radjax_student.students") for name in _imports(path)
            )
            for package in ("architecture", "runtime", "learning")
            for path in (source_root / package).rglob("*.py")
        )
    )
    legacy_ok = (
        "load_dense_tome_targets"
        not in (source_root / "artifacts" / "__init__.py").read_text(encoding="utf-8")
        and "run_tiny_train_step"
        not in (source_root / "training" / "__init__.py").read_text(encoding="utf-8")
        and (source_root / "legacy").is_dir()
    )
    hf_imports = _imports(source_root / "hf" / "contracts.py")
    hf_ok = not any(
        name.split(".", 1)[0] in {"transformers", "safetensors"} for name in hf_imports
    )
    checkpoint_source = (source_root / "checkpoints" / "learning.py").read_text(
        encoding="utf-8"
    )
    checkpoint_ok = all(
        token in checkpoint_source
        for token in (
            "CONTINUATION_CHECKPOINT_ROLE",
            "payload_descriptors",
            "source.json",
        )
    )
    docs = (root / "docs" / "P3_5_8_DOCUMENTATION_RECONCILIATION.md").read_text(
        encoding="utf-8"
    )
    documentation_ok = all(
        phrase in docs
        for phrase in ("production Tome metadata inspection", "Not yet proven", "P4.1")
    ) and "BLOCKED ON P3.9" not in (root / "docs" / "ROADMAP.md").read_text(
        encoding="utf-8"
    )
    phase1_ok = (
        json.loads((root / "phase1_acceptance_receipt.json").read_text())["status"]
        == "pass"
    )
    phase2_ok = (
        json.loads((root / "runtime_phase2_acceptance_receipt.json").read_text())[
            "status"
        ]
        == "pass"
    )
    from radjax_student.learning.p3_10_acceptance import (
        run_p3_10_learning_core_acceptance,
    )

    phase3_ok = run_p3_10_learning_core_acceptance().status == "pass"
    import_ok = all(
        not any(name.startswith("radjax_student.students") for name in _imports(path))
        for package in ("architecture", "runtime", "learning")
        for path in (source_root / package).rglob("*.py")
    )

    values = {
        "dependency_boundaries_valid": dependency_ok,
        "architecture_objective_separation_valid": architecture_ok,
        "jax_native_learning_contract_valid": jax_ok,
        "architecture_namespace_valid": namespace_ok,
        "legacy_isolation_valid": legacy_ok,
        "hf_preservation_contract_valid": hf_ok,
        "checkpoint_ownership_valid": checkpoint_ok,
        "documentation_consistency_valid": documentation_ok,
        "phase1_regression_valid": phase1_ok,
        "phase2_regression_valid": phase2_ok,
        "phase3_regression_valid": phase3_ok,
        "import_purity_valid": import_ok,
        "deterministic_replay_valid": replay_ok,
    }
    return P35ArchitectureIntegrityReceipt(
        schema_version=SCHEMA,
        status="pass" if all(values.values()) and not blockers else "fail",
        blockers=tuple(blockers),
        metadata={"gate": "P3.5.10", "module_count": audit["module_count"]},
        **values,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json"),
    )
    args = parser.parse_args(argv)
    receipt = run_p3_5_architecture_integrity_acceptance()
    args.output.write_text(receipt.to_json() + "\n", encoding="utf-8")
    print(f"P3.5.10 Architecture Integrity: {receipt.status}")
    return 0 if receipt.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
