"""P3.5 adversarial architecture-integrity acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.learning.errors import LearningIssue
from radjax_student.validation.architecture_audit import build_architecture_audit

SCHEMA = "radjax.p3_5_architecture_integrity_receipt.v2"
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
_FORBIDDEN_IMPORTS = {
    "accelerate",
    "datasets",
    "jax",
    "jaxlib",
    "radjax_tome",
    "torch",
    "transformers",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class GateCheck:
    valid: bool
    evidence: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    issue: LearningIssue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("gate checks must be boolean")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        if not self.valid and self.issue is None:
            raise ValueError("failed gate checks require a structured issue")


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
    blockers: tuple[LearningIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA or self.status not in {"pass", "fail"}:
            raise ValueError("invalid P3.5 receipt schema or status")
        flags = {name: getattr(self, name) for name in FLAGS}
        if any(type(value) is not bool for value in flags.values()):
            raise TypeError("P3.5 receipt flags must be booleans")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, LearningIssue) for item in blockers):
            raise TypeError("P3.5 receipt blockers must be LearningIssue values")
        expected_pass = all(flags.values()) and not blockers
        if (self.status == "pass") != expected_pass:
            raise ValueError("P3.5 receipt status does not match evidence")
        if self.status == "fail" and not blockers:
            raise ValueError("failed P3.5 receipt requires blocker evidence")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            **{name: getattr(self, name) for name in FLAGS},
            "blockers": [item.to_dict() for item in self.blockers],
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _issue(code: str, message: str, **details: Any) -> LearningIssue:
    return LearningIssue.create(code, message, **details)


def _passed(**evidence: Any) -> GateCheck:
    return GateCheck(True, evidence)


def _failed(code: str, message: str, **details: Any) -> GateCheck:
    return GateCheck(False, details, _issue(code, message, **details))


def _run_command(root: Path, target: str) -> GateCheck:
    command = (
        [sys.executable, "-m", "radjax_student.learning.p3_10_acceptance", "--json"]
        if target == "phase3_module"
        else [sys.executable, "-m", "pytest", "-q", target]
    )
    result = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    evidence = {"target": target, "returncode": result.returncode}
    if result.returncode:
        return _failed("learning_step_failed", "regression command failed", **evidence)
    return _passed(**evidence)


def _run_jax_contract() -> GateCheck:
    if importlib.util.find_spec("jax") is None:
        return _failed("learning_scope_capability_missing", "JAX is unavailable")
    try:
        import jax
        import jax.numpy as jnp

        from radjax_student.architecture import ForwardResult
        from radjax_student.learning import ObjectiveScope
        from radjax_student.learning.jax_core import (
            JaxBatch,
            JaxObjectiveConfig,
            build_jax_loss_fn,
            build_value_and_grad_fn,
            validate_finite_loss_and_gradients,
        )

        class LinearArchitecture:
            def apply_jax(
                self, parameters, carry, batch, *, objective_scope, training, rng_key
            ):
                del training, rng_key
                if not isinstance(objective_scope, ObjectiveScope):
                    raise TypeError("objective scope is not typed")
                output = (
                    jnp.dot(batch.inputs[:, None], parameters["weight"][None, :])
                    + parameters["bias"]
                )
                return ForwardResult(
                    outputs=output,
                    updated_architecture_carry={"step": carry["step"] + 1},
                )

        class Objective:
            def evaluate(self, surface, targets, weights, objective_config):
                del weights, objective_config
                return jnp.mean((surface - targets) ** 2), {"loss": jnp.asarray(1.0)}

        args = (
            {"weight": jnp.asarray((0.0,)), "bias": jnp.asarray((0.0,))},
            {"step": jnp.asarray(0.0)},
            JaxBatch(
                jnp.asarray((-1.0, 0.0, 1.0)), jnp.asarray(((-1.0,), (1.0,), (3.0,)))
            ),
            JaxObjectiveConfig("mse", ObjectiveScope()),
            jax.random.key(0),
        )
        value_and_grad = build_value_and_grad_fn(
            build_jax_loss_fn(LinearArchitecture(), Objective())
        )
        first = value_and_grad(*args)
        second = value_and_grad(*args)
        validate_finite_loss_and_gradients(first[0][0], first[1])
        replay = bool(jnp.array_equal(first[0][0], second[0][0])) and bool(
            jnp.array_equal(first[1]["weight"], second[1]["weight"])
        )
        if not replay:
            return _failed("learning_step_failed", "JAX replay diverged")
        return _passed(
            loss=float(first[0][0]),
            carry=float(first[0][1].updated_architecture_carry["step"]),
        )
    except Exception as exc:
        return _failed(
            "learning_internal_error",
            "JAX contract execution failed",
            exception=type(exc).__name__,
        )


def _check_namespace_and_legacy() -> tuple[GateCheck, GateCheck]:
    import radjax_student
    import radjax_student.steps as steps
    from radjax_student.architecture import ArchitectureRegistry

    namespace = (
        ArchitectureRegistry.__module__ == "radjax_student.architecture.registry"
        and "students" not in getattr(radjax_student, "__all__", ())
    )
    legacy = (
        not hasattr(steps, "learning_step")
        and not hasattr(steps, "ScalarObjective")
        and "step_executor" in inspect.signature(steps.run_learning_loop).parameters
    )
    return (
        _passed(registry=ArchitectureRegistry.__module__)
        if namespace
        else _failed(
            "learning_scope_capability_missing", "architecture namespace is ambiguous"
        ),
        _passed(explicit_executor=True)
        if legacy
        else _failed("learning_step_failed", "legacy scalar path remains a default"),
    )


def _check_hf() -> GateCheck:
    from radjax_student.architecture import (
        ArchitectureConfig,
        ParameterCatalog,
        ParameterDescriptor,
    )
    from radjax_student.hf import HFCompatibilityDescriptor, HFParameterMapping

    config = ArchitectureConfig("p35", model_config={"width": 1}, vocab_size=4)
    catalog = ParameterCatalog(
        "p35", (ParameterDescriptor("head.weight", (1,), "float32"),)
    )
    mapping = HFParameterMapping(
        "head.weight", "head/weight", "head.weight", (1,), "float32"
    )
    try:
        descriptor = HFCompatibilityDescriptor.from_architecture(
            config,
            catalog,
            model_type="p35",
            tokenizer_id="p35-tokenizer",
            special_token_ids={"pad": 0},
            parameter_mappings=(mapping,),
        )
        descriptor.validate_against(config, catalog)
    except Exception as exc:
        return _failed(
            "learning_step_failed",
            "HF preservation contract failed",
            exception=type(exc).__name__,
        )
    return _passed(mapping_count=len(descriptor.parameter_mappings))


def _check_checkpoint() -> GateCheck:
    from radjax_student.checkpoints import (
        CheckpointPayloadDescriptor,
        FutureTensorPayloadDescriptor,
    )

    current = CheckpointPayloadDescriptor(
        "architecture", "json", "scalar_parameter_mapping"
    )
    future = FutureTensorPayloadDescriptor("jax_pytree.v1", "future_codec")
    if (
        current.kind != "scalar_parameter_mapping"
        or future.storage_codec != "future_codec"
    ):
        return _failed("learning_step_failed", "checkpoint ownership is inaccurate")
    return _passed(current=current.to_dict(), future=future.to_dict())


def _check_docs(root: Path) -> GateCheck:
    text = (root / "docs" / "P3_5_8_DOCUMENTATION_RECONCILIATION.md").read_text(
        encoding="utf-8"
    )
    required = ("pure JAX", "legacy", "Not yet proven", "P4.1")
    if not all(item.lower() in text.lower() for item in required):
        return _failed("learning_step_failed", "documentation is stale")
    return _passed(document="P3_5_8_DOCUMENTATION_RECONCILIATION.md")


def _check_import_purity(root: Path) -> GateCheck:
    for module in (
        "radjax_student",
        "radjax_student.architecture",
        "radjax_student.learning",
        "radjax_student.steps",
    ):
        script = (
            "import importlib, sys; importlib.import_module(" + repr(module) + "); "
            "print('\\n'.join(sorted({name.split('.', 1)[0] for name in sys.modules})))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        loaded = set(result.stdout.splitlines())
        leaked = sorted(loaded & _FORBIDDEN_IMPORTS)
        if result.returncode or leaked:
            return _failed(
                "learning_step_failed",
                "base import purity failed",
                module=module,
                leaked=leaked,
            )
    return _passed(modules=("radjax_student", "architecture", "learning", "steps"))


@dataclass(frozen=True)
class P35AcceptanceDependencies:
    build_audit: Callable[[Path], Mapping[str, Any]] = build_architecture_audit
    run_jax_contract: Callable[[], GateCheck] = _run_jax_contract
    check_namespace_legacy: Callable[[], tuple[GateCheck, GateCheck]] = (
        _check_namespace_and_legacy
    )
    check_hf: Callable[[], GateCheck] = _check_hf
    check_checkpoint: Callable[[], GateCheck] = _check_checkpoint
    check_docs: Callable[[Path], GateCheck] = _check_docs
    run_phase1: Callable[[Path, str], GateCheck] = _run_command
    run_phase2: Callable[[Path, str], GateCheck] = _run_command
    run_phase3: Callable[[Path, str], GateCheck] = _run_command
    check_import_purity: Callable[[Path], GateCheck] = _check_import_purity


def run_p3_5_architecture_integrity_acceptance(
    repo_root: Path | None = None,
    *,
    dependencies: P35AcceptanceDependencies | None = None,
) -> P35ArchitectureIntegrityReceipt:
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    deps = dependencies or P35AcceptanceDependencies()
    try:
        audit = deps.build_audit(root)
        dependency = (
            _passed(
                audit_digest=_digest(audit), module_count=audit.get("module_count", 0)
            )
            if audit.get("status") == "pass" and not audit.get("blockers")
            else _failed(
                "learning_step_failed",
                "dependency audit failed",
                blockers=audit.get("blockers", []),
            )
        )
        architecture = (
            _passed(audit_digest=_digest(audit))
            if dependency.valid
            else GateCheck(False, dependency.evidence, dependency.issue)
        )
        namespace, legacy = deps.check_namespace_legacy()
        hf = deps.check_hf()
        checkpoint = deps.check_checkpoint()
        docs = deps.check_docs(root)
        phase1 = deps.run_phase1(root, "tests/acceptance")
        phase2 = deps.run_phase2(root, "tests/acceptance/runtime")
        phase3 = deps.run_phase3(root, "phase3_module")
        imports = deps.check_import_purity(root)
        jax_check = deps.run_jax_contract()
        checks = {
            "dependency_boundaries_valid": dependency,
            "architecture_objective_separation_valid": architecture,
            "jax_native_learning_contract_valid": jax_check,
            "architecture_namespace_valid": namespace,
            "legacy_isolation_valid": legacy,
            "hf_preservation_contract_valid": hf,
            "checkpoint_ownership_valid": checkpoint,
            "documentation_consistency_valid": docs,
            "phase1_regression_valid": phase1,
            "phase2_regression_valid": phase2,
            "phase3_regression_valid": phase3,
            "import_purity_valid": imports,
        }
        replay_evidence = {name: dict(check.evidence) for name, check in checks.items()}
        replay = _passed(evidence_digest=_digest(replay_evidence))
        checks["deterministic_replay_valid"] = replay
        blockers = tuple(
            check.issue for check in checks.values() if check.issue is not None
        )
        values = {name: check.valid for name, check in checks.items()}
        return P35ArchitectureIntegrityReceipt(
            schema_version=SCHEMA,
            status="pass" if all(values.values()) else "fail",
            blockers=blockers,
            metadata={"gate": "P3.5.10A", "evidence_digest": _digest(replay_evidence)},
            **values,
        )
    except Exception as exc:
        issue = _issue(
            "learning_internal_error",
            "P3.5 gate internal error",
            exception=type(exc).__name__,
        )
        values = {name: False for name in FLAGS}
        return P35ArchitectureIntegrityReceipt(
            schema_version=SCHEMA,
            status="fail",
            blockers=(issue,),
            metadata={"gate": "P3.5.10A"},
            **values,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check-recorded", action="store_true")
    parser.add_argument("--write-receipt", type=Path)
    args = parser.parse_args(argv)
    receipt = run_p3_5_architecture_integrity_acceptance()
    if args.write_receipt:
        args.write_receipt.write_text(receipt.to_json() + "\n", encoding="utf-8")
    recorded_matches = True
    if args.check_recorded:
        path = (
            Path(__file__).resolve().parents[3]
            / "docs"
            / "P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json"
        )
        recorded_matches = (
            path.is_file()
            and path.read_text(encoding="utf-8").strip() == receipt.to_json()
        )
    if args.json:
        print(receipt.to_json())
    else:
        print(f"P3.5.10A Architecture Integrity: {receipt.status}")
    return 0 if receipt.status == "pass" and recorded_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
