"""P3.5 architecture-integrity acceptance with replayed behavioral evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.learning.errors import LearningIssue
from radjax_student.validation.architecture_audit import build_architecture_audit

SCHEMA = "radjax.p3_5_architecture_integrity_receipt.v3"
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
SECTION_CODES = {
    "dependency_boundaries_valid": "p35_dependency_boundaries_failed",
    "architecture_objective_separation_valid": (
        "p35_architecture_objective_separation_failed"
    ),
    "jax_native_learning_contract_valid": "p35_jax_native_learning_contract_failed",
    "architecture_namespace_valid": "p35_architecture_namespace_failed",
    "legacy_isolation_valid": "p35_legacy_isolation_failed",
    "hf_preservation_contract_valid": "p35_hf_preservation_contract_failed",
    "checkpoint_ownership_valid": "p35_checkpoint_ownership_failed",
    "documentation_consistency_valid": "p35_documentation_consistency_failed",
    "phase1_regression_valid": "p35_phase1_regression_failed",
    "phase2_regression_valid": "p35_phase2_regression_failed",
    "phase3_regression_valid": "p35_phase3_regression_failed",
    "import_purity_valid": "p35_import_purity_failed",
    "deterministic_replay_valid": "p35_deterministic_replay_failed",
}
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
        if type(self.valid) is not bool:
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
        values = {name: getattr(self, name) for name in FLAGS}
        if any(type(value) is not bool for value in values.values()):
            raise TypeError("P3.5 receipt flags must be booleans")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, LearningIssue) for item in blockers):
            raise TypeError("P3.5 receipt blockers must be LearningIssue values")
        expected_pass = all(values.values()) and not blockers
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


def _check_failure(section: str, message: str, **details: Any) -> GateCheck:
    return GateCheck(
        False,
        details,
        LearningIssue.create(SECTION_CODES[section], message, **details),
    )


def _check_success(**evidence: Any) -> GateCheck:
    return GateCheck(True, evidence)


def _expect_rejected(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except Exception:
        return True
    return False


def _run_command(root: Path, target: str, section: str) -> GateCheck:
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
        return _check_failure(section, "regression command failed", **evidence)
    return _check_success(**evidence)


def _check_architecture_objective() -> GateCheck:
    if importlib.util.find_spec("jax") is None:
        return _check_failure(
            "architecture_objective_separation_valid", "JAX is unavailable"
        )
    try:
        import jax
        import jax.numpy as jnp

        from radjax_student.architecture import ForwardResult
        from radjax_student.learning import ObjectiveScope
        from radjax_student.learning.jax_core import (
            JaxBatch,
            JaxObjectiveConfig,
            build_jax_loss_fn,
        )

        class Architecture:
            def apply_jax(
                self, parameters, carry, batch, *, objective_scope, training, rng_key
            ):
                del training, rng_key
                if not isinstance(objective_scope, ObjectiveScope):
                    raise TypeError("objective scope must be typed")
                output = jnp.dot(batch.inputs[:, None], parameters["weight"][None, :])
                return ForwardResult(
                    outputs=output,
                    surface_values={"hidden": output + 1.0},
                    updated_architecture_carry={"carry": carry["carry"] + 1.0},
                )

        class Objective:
            def __init__(self):
                self.surfaces: list[Any] = []

            def evaluate(self, surface, targets, weights, objective_config):
                del weights, objective_config
                if isinstance(surface, Mapping):
                    raise TypeError("objective received a parameter mapping")
                self.surfaces.append(surface)
                return jnp.mean((surface - targets) ** 2), {}

        architecture = Architecture()
        objective = Objective()
        loss_fn = build_jax_loss_fn(architecture, objective)
        batch = JaxBatch(
            jnp.asarray((-1.0, 0.0, 1.0)), jnp.asarray(((-1.0,), (1.0,), (3.0,)))
        )
        parameters = {"weight": jnp.asarray((1.0,))}
        carry = {"carry": jnp.asarray(0.0)}
        config = JaxObjectiveConfig("mse", ObjectiveScope())
        baseline, _ = loss_fn(parameters, carry, batch, config, None)
        altered, _ = loss_fn(
            {"weight": jnp.asarray((2.0,))}, carry, batch, config, None
        )
        missing = JaxObjectiveConfig(
            "mse", ObjectiveScope("intermediate_surface", "missing")
        )
        missing_rejected = _expect_rejected(
            lambda: loss_fn(parameters, carry, batch, missing, None)
        )
        jaxpr = jax.make_jaxpr(loss_fn)(parameters, carry, batch, config, None)
        dots = sum(item.primitive.name == "dot_general" for item in jaxpr.jaxpr.eqns)
        if (
            not objective.surfaces
            or float(baseline) == float(altered)
            or not missing_rejected
            or dots != 1
        ):
            return _check_failure(
                "architecture_objective_separation_valid",
                "architecture/objective behavioral contract failed",
                forward_dots=dots,
                missing_rejected=missing_rejected,
            )
        return _check_success(
            objective_surface_only=True,
            loss_changes_with_architecture=True,
            missing_surface_rejected=True,
            forward_dots=dots,
        )
    except Exception as exc:
        return _check_failure(
            "architecture_objective_separation_valid",
            "architecture/objective check raised",
            exception=type(exc).__name__,
        )


def _run_jax_contract() -> GateCheck:
    if importlib.util.find_spec("jax") is None:
        return _check_failure(
            "jax_native_learning_contract_valid", "JAX is unavailable"
        )
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
                    updated_architecture_carry={"step": carry["step"] + 1.0},
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
        loss_fn = build_jax_loss_fn(LinearArchitecture(), Objective())
        value_and_grad = build_value_and_grad_fn(loss_fn)
        first = value_and_grad(*args)
        second = value_and_grad(*args)
        validate_finite_loss_and_gradients(first[0][0], first[1])
        nonfinite_loss_rejected = _expect_rejected(
            lambda: validate_finite_loss_and_gradients(jnp.asarray(jnp.nan), first[1])
        )
        nonfinite_gradient_rejected = _expect_rejected(
            lambda: validate_finite_loss_and_gradients(
                first[0][0], {"x": jnp.asarray(jnp.inf)}
            )
        )

        def output_carry(input_carry):
            return loss_fn(args[0], {"step": input_carry}, args[2], args[3], args[4])[
                1
            ].updated_architecture_carry["step"]

        carry_gradient = float(jax.grad(output_carry)(jnp.asarray(3.0)))
        replay = bool(jnp.array_equal(first[0][0], second[0][0])) and bool(
            jnp.array_equal(first[1]["weight"], second[1]["weight"])
        )
        if not all(
            (
                replay,
                nonfinite_loss_rejected,
                nonfinite_gradient_rejected,
                carry_gradient == 0.0,
            )
        ):
            return _check_failure(
                "jax_native_learning_contract_valid",
                "JAX contract behavioral proof failed",
                replay=replay,
                nonfinite_loss_rejected=nonfinite_loss_rejected,
                nonfinite_gradient_rejected=nonfinite_gradient_rejected,
                carry_gradient=carry_gradient,
            )
        return _check_success(
            loss=float(first[0][0]),
            replay=replay,
            nonfinite_loss_rejected=nonfinite_loss_rejected,
            nonfinite_gradient_rejected=nonfinite_gradient_rejected,
            output_carry_gradient=carry_gradient,
        )
    except Exception as exc:
        return _check_failure(
            "jax_native_learning_contract_valid",
            "JAX contract execution failed",
            exception=type(exc).__name__,
        )


def _check_namespace_and_legacy() -> tuple[GateCheck, GateCheck]:
    import radjax_student
    import radjax_student.steps as steps
    from radjax_student.architecture import ArchitectureRegistry

    namespace_ok = (
        ArchitectureRegistry.__module__ == "radjax_student.architecture.registry"
        and "students" not in getattr(radjax_student, "__all__", ())
    )
    legacy_ok = (
        not hasattr(steps, "learning_step")
        and not hasattr(steps, "ScalarObjective")
        and "step_executor" in inspect.signature(steps.run_learning_loop).parameters
    )
    namespace = (
        _check_success(registry=ArchitectureRegistry.__module__)
        if namespace_ok
        else _check_failure(
            "architecture_namespace_valid", "architecture namespace is ambiguous"
        )
    )
    legacy = (
        _check_success(explicit_executor=True)
        if legacy_ok
        else _check_failure(
            "legacy_isolation_valid", "legacy scalar path remains a default"
        )
    )
    return namespace, legacy


def _hf_fixture():
    from radjax_student.architecture import (
        ArchitectureConfig,
        ParameterCatalog,
        ParameterDescriptor,
    )
    from radjax_student.hf import HFParameterMapping

    config = ArchitectureConfig("p35", model_config={"width": 1}, vocab_size=4)
    catalog = ParameterCatalog(
        "p35",
        (
            ParameterDescriptor("head.bias", (1,), "float32"),
            ParameterDescriptor("head.weight", (1, 1), "float32"),
        ),
    )
    mappings = (
        HFParameterMapping("head.bias", "head/bias", "head.bias", (1,), "float32"),
        HFParameterMapping(
            "head.weight", "head/weight", "head.weight", (1, 1), "float32"
        ),
    )
    return config, catalog, mappings


def _check_hf() -> GateCheck:
    from radjax_student.architecture import ArchitectureConfig
    from radjax_student.hf import HFCompatibilityDescriptor, HFParameterMapping

    config, catalog, mappings = _hf_fixture()
    try:
        descriptor = HFCompatibilityDescriptor.from_architecture(
            config,
            catalog,
            model_type="p35",
            tokenizer_id="p35-tokenizer",
            special_token_ids={"pad": 0},
            parameter_mappings=mappings,
        )
        duplicate_jax = (
            mappings[0],
            HFParameterMapping(
                "head.weight", "head/bias", "head.weight", (1, 1), "float32"
            ),
        )
        duplicate_hf = (
            mappings[0],
            HFParameterMapping(
                "head.weight", "head/weight", "head.bias", (1, 1), "float32"
            ),
        )
        checks = {
            "duplicate_jax_rejected": _expect_rejected(
                lambda: HFCompatibilityDescriptor.from_architecture(
                    config,
                    catalog,
                    model_type="p35",
                    tokenizer_id="p35",
                    special_token_ids={"pad": 0},
                    parameter_mappings=duplicate_jax,
                )
            ),
            "duplicate_hf_rejected": _expect_rejected(
                lambda: HFCompatibilityDescriptor.from_architecture(
                    config,
                    catalog,
                    model_type="p35",
                    tokenizer_id="p35",
                    special_token_ids={"pad": 0},
                    parameter_mappings=duplicate_hf,
                )
            ),
            "missing_catalog_rejected": _expect_rejected(
                lambda: HFCompatibilityDescriptor.from_architecture(
                    config,
                    catalog,
                    model_type="p35",
                    tokenizer_id="p35",
                    special_token_ids={"pad": 0},
                    parameter_mappings=(mappings[0],),
                )
            ),
            "shape_mismatch_rejected": _expect_rejected(
                lambda: HFCompatibilityDescriptor.from_architecture(
                    config,
                    catalog,
                    model_type="p35",
                    tokenizer_id="p35",
                    special_token_ids={"pad": 0},
                    parameter_mappings=(
                        mappings[0],
                        HFParameterMapping(
                            "head.weight",
                            "head/weight",
                            "head.weight",
                            (2, 1),
                            "float32",
                        ),
                    ),
                )
            ),
            "runtime_name_rejected": _expect_rejected(
                lambda: HFParameterMapping(
                    "head.weight", "mesh/weight", "head.weight", (1, 1), "float32"
                )
            ),
            "config_conflict_rejected": _expect_rejected(
                lambda: descriptor.validate_against(
                    ArchitectureConfig("p35", model_config={"width": 2}, vocab_size=4),
                    catalog,
                )
            ),
        }
        imports = ast.parse(
            Path(
                __import__(
                    "radjax_student.hf.contracts", fromlist=["__file__"]
                ).__file__
            ).read_text()
        )
        optional_import = any(
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name.split(".", 1)[0] in {"transformers", "safetensors"}
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".", 1)[0]
                in {"transformers", "safetensors"}
            )
            for node in ast.walk(imports)
        )
        if not all(checks.values()) or optional_import:
            return _check_failure(
                "hf_preservation_contract_valid",
                "HF preservation adversarial checks failed",
                **checks,
                optional_import=optional_import,
            )
        return _check_success(
            mapping_count=len(descriptor.parameter_mappings),
            **checks,
            optional_import=optional_import,
        )
    except Exception as exc:
        return _check_failure(
            "hf_preservation_contract_valid",
            "HF preservation check raised",
            exception=type(exc).__name__,
        )


def _checkpoint_fixture():
    from radjax_student.architecture import ArchitectureState
    from radjax_student.checkpoints import LearningCheckpoint
    from radjax_student.learning import LearningState
    from radjax_student.optimizers import OptimizerState

    return LearningCheckpoint(
        "p35-runtime",
        LearningState("p35", global_step=1, optimizer_step=1),
        ArchitectureState("p35.arch"),
        OptimizerState("sgd.v1", ("head.weight",), step=1, backend_state={"step": 1}),
        {"head.weight": 0.5},
        {"position": 1},
        {},
        {},
    )


def _check_checkpoint() -> GateCheck:
    from radjax_student.checkpoints import (
        HF_DISTRIBUTION_CHECKPOINT_ROLE,
        CheckpointPayloadDescriptor,
        FutureTensorPayloadDescriptor,
        LearningCheckpoint,
        load_learning_checkpoint,
        reject_implicit_hf_conversion,
        save_learning_checkpoint,
    )

    try:
        checkpoint = _checkpoint_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_learning_checkpoint(checkpoint, directory)
            loaded = load_learning_checkpoint(
                directory, runtime_reference="p35-runtime"
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            source = (directory / "source.json").read_bytes()
            (directory / "source.json").write_bytes(source + b" ")
            integrity_rejected = _expect_rejected(
                lambda: load_learning_checkpoint(directory)
            )
        role_rejected = _expect_rejected(
            lambda: reject_implicit_hf_conversion(checkpoint)
        )
        distribution_rejected = _expect_rejected(
            lambda: LearningCheckpoint(
                checkpoint.runtime_reference,
                checkpoint.learning_state,
                checkpoint.architecture_state,
                checkpoint.optimizer_state,
                checkpoint.parameters,
                checkpoint.source_state,
                {},
                {},
                role=HF_DISTRIBUTION_CHECKPOINT_ROLE,
            )
        )
        descriptor = manifest["payload_descriptors"]["architecture.json"]
        checks = {
            "round_trip": loaded.source_state == checkpoint.source_state,
            "role_rejected": role_rejected and distribution_rejected,
            "source_integrity_rejected": integrity_rejected,
            "scalar_descriptor": descriptor["kind"] == "scalar_parameter_mapping",
            "tensor_not_emitted": all(
                "tensor" not in item["kind"]
                for item in manifest["payload_descriptors"].values()
            ),
            "no_runtime_handles": "runtime_handle"
            not in json.dumps(manifest, sort_keys=True),
            "typed_descriptors": CheckpointPayloadDescriptor.from_dict(descriptor).kind
            == "scalar_parameter_mapping"
            and FutureTensorPayloadDescriptor(
                "jax_pytree.v1", "future_codec"
            ).storage_codec
            == "future_codec",
        }
        if not all(checks.values()):
            return _check_failure(
                "checkpoint_ownership_valid",
                "checkpoint ownership adversarial checks failed",
                **checks,
            )
        return _check_success(**checks)
    except Exception as exc:
        return _check_failure(
            "checkpoint_ownership_valid",
            "checkpoint ownership check raised",
            exception=type(exc).__name__,
        )


def _check_docs(root: Path) -> GateCheck:
    text = (root / "docs" / "P3_5_8_DOCUMENTATION_RECONCILIATION.md").read_text(
        encoding="utf-8"
    )
    required = ("pure JAX", "legacy", "Not yet proven", "P4.1")
    if not all(item.lower() in text.lower() for item in required):
        return _check_failure(
            "documentation_consistency_valid", "documentation is stale"
        )
    return _check_success(document="P3_5_8_DOCUMENTATION_RECONCILIATION.md")


def _check_import_purity(root: Path) -> GateCheck:
    for module in (
        "radjax_student",
        "radjax_student.architecture",
        "radjax_student.learning",
        "radjax_student.steps",
    ):
        script = (
            "import importlib, sys; importlib.import_module("
            + repr(module)
            + "); print('\\n'.join(sorted("
            + "{name.split('.', 1)[0] for name in sys.modules})))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        leaked = sorted(set(result.stdout.splitlines()) & _FORBIDDEN_IMPORTS)
        if result.returncode or leaked:
            return _check_failure(
                "import_purity_valid",
                "base import purity failed",
                module=module,
                leaked=leaked,
            )
    return _check_success(
        modules=("radjax_student", "architecture", "learning", "steps")
    )


@dataclass(frozen=True)
class P35AcceptanceDependencies:
    build_audit: Callable[[Path], Mapping[str, Any]] = build_architecture_audit
    check_architecture_objective: Callable[[], GateCheck] = (
        _check_architecture_objective
    )
    run_jax_contract: Callable[[], GateCheck] = _run_jax_contract
    check_namespace_legacy: Callable[[], tuple[GateCheck, GateCheck]] = (
        _check_namespace_and_legacy
    )
    check_hf: Callable[[], GateCheck] = _check_hf
    check_checkpoint: Callable[[], GateCheck] = _check_checkpoint
    check_docs: Callable[[Path], GateCheck] = _check_docs
    run_command: Callable[[Path, str, str], GateCheck] = _run_command
    check_import_purity: Callable[[Path], GateCheck] = _check_import_purity


def _collect_checks(
    root: Path, deps: P35AcceptanceDependencies
) -> dict[str, GateCheck]:
    audit = deps.build_audit(root)
    dependency = (
        _check_success(
            audit_digest=_digest(audit), module_count=audit.get("module_count", 0)
        )
        if audit.get("status") == "pass" and not audit.get("blockers")
        else _check_failure(
            "dependency_boundaries_valid",
            "dependency audit failed",
            blockers=audit.get("blockers", []),
        )
    )
    namespace, legacy = deps.check_namespace_legacy()
    return {
        "dependency_boundaries_valid": dependency,
        "architecture_objective_separation_valid": deps.check_architecture_objective(),
        "jax_native_learning_contract_valid": deps.run_jax_contract(),
        "architecture_namespace_valid": namespace,
        "legacy_isolation_valid": legacy,
        "hf_preservation_contract_valid": deps.check_hf(),
        "checkpoint_ownership_valid": deps.check_checkpoint(),
        "documentation_consistency_valid": deps.check_docs(root),
        "phase1_regression_valid": deps.run_command(
            root, "tests/acceptance", "phase1_regression_valid"
        ),
        "phase2_regression_valid": deps.run_command(
            root, "tests/acceptance/runtime", "phase2_regression_valid"
        ),
        "phase3_regression_valid": deps.run_command(
            root, "phase3_module", "phase3_regression_valid"
        ),
        "import_purity_valid": deps.check_import_purity(root),
    }


def _canonical_evidence(checks: Mapping[str, GateCheck]) -> dict[str, Any]:
    return {
        name: {
            "valid": check.valid,
            "evidence": dict(check.evidence),
            "issue": None if check.issue is None else check.issue.to_dict(),
        }
        for name, check in sorted(checks.items())
    }


def run_p3_5_architecture_integrity_acceptance(
    repo_root: Path | None = None,
    *,
    dependencies: P35AcceptanceDependencies | None = None,
) -> P35ArchitectureIntegrityReceipt:
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    deps = dependencies or P35AcceptanceDependencies()
    try:
        first = _collect_checks(root, deps)
        second = _collect_checks(root, deps)
        first_evidence, second_evidence = (
            _canonical_evidence(first),
            _canonical_evidence(second),
        )
        replay = (
            _check_success(evidence_digest=_digest(first_evidence))
            if first_evidence == second_evidence
            else _check_failure(
                "deterministic_replay_valid",
                "complete P3.5 evidence replay diverged",
                first_digest=_digest(first_evidence),
                second_digest=_digest(second_evidence),
            )
        )
        checks = {**first, "deterministic_replay_valid": replay}
        blockers = tuple(
            check.issue for check in checks.values() if check.issue is not None
        )
        values = {name: check.valid for name, check in checks.items()}
        return P35ArchitectureIntegrityReceipt(
            schema_version=SCHEMA,
            status="pass" if all(values.values()) else "fail",
            blockers=blockers,
            metadata={
                "gate": "P3.5.10A",
                "first_evidence_digest": _digest(first_evidence),
                "second_evidence_digest": _digest(second_evidence),
            },
            **values,
        )
    except Exception as exc:
        issue = LearningIssue.create(
            "p35_internal_error",
            "P3.5 gate internal error",
            exception=type(exc).__name__,
        )
        values = {name: False for name in FLAGS}
        return P35ArchitectureIntegrityReceipt(
            SCHEMA, "fail", blockers=(issue,), metadata={"gate": "P3.5.10A"}, **values
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
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json"
    )
    recorded_matches = not args.check_recorded or (
        path.is_file() and path.read_text(encoding="utf-8").strip() == receipt.to_json()
    )
    print(
        receipt.to_json()
        if args.json
        else f"P3.5.10A Architecture Integrity: {receipt.status}"
    )
    return 0 if receipt.status == "pass" and recorded_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
