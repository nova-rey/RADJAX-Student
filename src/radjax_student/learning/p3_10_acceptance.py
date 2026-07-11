"""P3.10 public-API golden acceptance gate for the generic learning core."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, Literal

from radjax_student import architecture as _architecture
from radjax_student.checkpoints import (
    LearningCheckpoint,
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from radjax_student.checkpoints.learning import CHECKPOINT_SCHEMA_VERSION
from radjax_student.learning import (
    LearningBatch,
    LearningConfig,
    LearningIssue,
    LearningState,
    ObjectiveRequest,
    ObjectiveResult,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
    canonical_learning_json,
    canonical_objective_json,
    run_p3_8_observability_acceptance,
)
from radjax_student.learning.synthetic_smoke import (
    P39SyntheticLearningReceipt,
    run_p3_9_synthetic_learning_smoke,
)
from radjax_student.optimizers import (
    GradientTree,
    OptimizerConfig,
    OptimizerInitRequest,
    OptimizerState,
    OptimizerUpdateRequest,
    SgdOptimizer,
)

ParameterCatalog = _architecture.ParameterCatalog
ParameterDescriptor = _architecture.ParameterDescriptor

SCHEMA = "radjax.p3_10_learning_core_acceptance.v1"
CLAIMS = (
    "p3_1_learning_contract_validated",
    "p3_2_architecture_boundary_validated",
    "p3_3_optimizer_validated",
    "p3_4_batch_objective_validated",
    "p3_5_single_step_validated",
    "p3_6_checkpoint_validated",
    "p3_7_learning_loop_validated",
    "p3_8_observability_validated",
    "p3_9_synthetic_learning_validated",
    "p3_10_learning_core_golden_gate_validated",
)
NON_CLAIMS = (
    "model_quality",
    "real_architecture_support",
    "tome_training",
    "language_modeling",
    "distributed_training",
    "accelerator_performance",
    "production_hyperparameters",
    "evaluation",
    "generalization",
)
VALIDITY_FIELDS = (
    "contracts_valid",
    "optimizer_valid",
    "single_step_valid",
    "loop_valid",
    "checkpoint_valid",
    "resume_valid",
    "observability_valid",
    "synthetic_learning_valid",
    "deterministic_replay_valid",
    "documentation_valid",
    "test_inventory_valid",
)
SECTION_CODES = {
    field_name: f"p3_10_{field_name.removesuffix('_valid')}_failed"
    for field_name in VALIDITY_FIELDS
}


@dataclass(frozen=True)
class P310AcceptanceDependencies:
    contracts_fn: Callable[[], Mapping[str, Any]]
    optimizer_fn: Callable[[], Mapping[str, Any]]
    single_step_fn: Callable[[], Mapping[str, Any]]
    loop_fn: Callable[[], Mapping[str, Any]]
    checkpoint_fn: Callable[[], Mapping[str, Any]]
    resume_fn: Callable[[], P39SyntheticLearningReceipt]
    observability_fn: Callable[[], Any]
    synthetic_fn: Callable[[], P39SyntheticLearningReceipt]
    replay_fn: Callable[
        [], tuple[P39SyntheticLearningReceipt, P39SyntheticLearningReceipt]
    ]
    documentation_fn: Callable[[], Mapping[str, Any]]
    test_inventory_fn: Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class P310LearningCoreAcceptanceReceipt:
    schema_version: str
    status: Literal["pass", "fail"]
    contracts_valid: bool
    optimizer_valid: bool
    single_step_valid: bool
    loop_valid: bool
    checkpoint_valid: bool
    resume_valid: bool
    observability_valid: bool
    synthetic_learning_valid: bool
    deterministic_replay_valid: bool
    documentation_valid: bool
    test_inventory_valid: bool
    blockers: tuple[LearningIssue, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    claims_made: tuple[str, ...] = CLAIMS
    claims_not_made: tuple[str, ...] = NON_CLAIMS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA or self.status not in {"pass", "fail"}:
            raise ValueError("invalid P3.10 acceptance receipt")
        flags = {name: getattr(self, name) for name in VALIDITY_FIELDS}
        if any(type(value) is not bool for value in flags.values()):
            raise TypeError("P3.10 validity flags must be booleans")
        blockers = tuple(self.blockers)
        warnings = tuple(self.warnings)
        if any(
            not isinstance(issue, LearningIssue) for issue in (*blockers, *warnings)
        ):
            raise TypeError("P3.10 findings must be LearningIssue values")
        passing = all(flags.values()) and not blockers
        if (self.status == "pass") != passing:
            raise ValueError("P3.10 status does not match validity evidence")
        if tuple(self.claims_made) != CLAIMS or len(set(self.claims_made)) != len(
            CLAIMS
        ):
            raise ValueError("P3.10 claims are invalid")
        if not set(NON_CLAIMS).issubset(self.claims_not_made):
            raise ValueError("P3.10 non-claims are incomplete")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("P3.10 metadata must be a mapping")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "claims_made", tuple(self.claims_made))
        object.__setattr__(self, "claims_not_made", tuple(self.claims_not_made))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            **{name: getattr(self, name) for name in VALIDITY_FIELDS},
            "blockers": [issue.to_dict() for issue in self.blockers],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "claims_made": list(self.claims_made),
            "claims_not_made": list(self.claims_not_made),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _contracts() -> Mapping[str, Any]:
    config = LearningConfig(
        update_scope=UpdateScope("whole_student"),
        objective_scope=ObjectiveScope("final_output"),
    )
    batch = LearningBatch("p310-batch", {"x": (1,)}, {"y": (2,)})
    request = ObjectiveRequest("p310-objective", batch_reference=batch.batch_id)
    result = ObjectiveResult("p310-objective", 1.0, {"mse": 1.0})
    return {
        "config": LearningConfig.from_dict(config.to_dict()).to_dict(),
        "batch": LearningBatch.from_dict(batch.to_dict()).to_dict(),
        "objective_request": ObjectiveRequest.from_dict(request.to_dict()).to_dict(),
        "objective_result": ObjectiveResult.from_dict(result.to_dict()).to_dict(),
        "learning_json": canonical_learning_json(config.to_dict()),
        "objective_json": canonical_objective_json(request.to_dict()),
    }


def _optimizer() -> Mapping[str, Any]:
    catalog = ParameterCatalog(
        "p310.synthetic",
        (
            ParameterDescriptor("head.bias", (), "float64", "output_head"),
            ParameterDescriptor("trunk.weight", (), "float64", "recurrent_block"),
        ),
    )
    selection = ResolvedUpdateSelection("p310:whole", catalog.paths, ())
    config = OptimizerConfig("sgd.v1", learning_rate=0.1)
    optimizer = SgdOptimizer()
    state = optimizer.initialize_state(
        OptimizerInitRequest(config, catalog, selection)
    ).optimizer_state
    update = optimizer.apply_updates(
        OptimizerUpdateRequest(
            GradientTree(catalog.paths, values={path: 1.0 for path in catalog.paths}),
            state,
            config,
            selection,
            0,
            parameters={path: 0.0 for path in catalog.paths},
        )
    )
    return {
        "optimizer_id": optimizer.optimizer_id,
        "step": update.updated_optimizer_state.step,
        "changed": update.changed_parameter_paths,
        "parameters": update.updated_parameters,
    }


def _checkpoint() -> Mapping[str, Any]:
    state = OptimizerState(
        "sgd.v1", ("p310.weight",), step=1, backend_state={"steps": 1}
    )
    checkpoint = LearningCheckpoint(
        "p310-runtime",
        LearningState(run_id="p310", optimizer_step=1, global_step=1),
        None,
        state,
        {"p310.weight": 0.9},
        {"source_id": "p310.source", "position": 1},
        {},
        {},
    )
    with TemporaryDirectory() as raw:
        path = Path(raw)
        saved = save_learning_checkpoint(checkpoint, path)
        loaded = load_learning_checkpoint(path, runtime_reference="p310-runtime")
        manifest = json.loads((path / "manifest.json").read_text())
    return {
        "schema": saved.schema_version,
        "source_state": loaded.source_state,
        "source_owned": manifest["ownership"]["source.json"] == "batch_source",
        "source_hashed": "source.json" in manifest["hashes"],
        "source_sized": "source.json" in manifest["sizes"],
    }


def _default_documentation() -> Mapping[str, Any]:
    root = Path(__file__).parents[3]
    docs = root / "docs"
    p36 = (docs / "P3_6_MODEL_AND_OPTIMIZER_CHECKPOINT_CONTRACT.md").read_text()
    p39 = (docs / "P3_9_SYNTHETIC_END_TO_END_LEARNING_SMOKE.md").read_text()
    index = (docs / "INDEX.md").read_text()
    readme = (root / "README.md").read_text()
    return {
        "p36_v2": "learning_checkpoint.v2" in p36 and "batch_source" in p36,
        "p39_resume": "restore" in p39.lower() and "source.json" in p39,
        "index": "P3.9 Synthetic End-to-End Learning Smoke" in index,
        "readme": "learning_checkpoint.v2" in readme,
    }


def _default_inventory() -> Mapping[str, Any]:
    root = Path(__file__).parents[3]
    files = (
        root / "tests/test_learning_checkpoint.py",
        root / "tests/test_p3_8_observability_acceptance.py",
        root / "tests/test_p3_9_synthetic_learning_smoke.py",
    )
    count = sum(
        text.count("\ndef test_") for text in (path.read_text() for path in files)
    )
    return {"named_tests": count, "required": 50, "files": len(files)}


def _default_single_step() -> Mapping[str, Any]:
    receipt = run_p3_9_synthetic_learning_smoke()
    return {"status": receipt.status, "loss_decrease": receipt.loss_decrease_valid}


def _default_loop() -> Mapping[str, Any]:
    receipt = run_p3_9_synthetic_learning_smoke()
    return {
        "status": receipt.status,
        "steps": receipt.whole_student.steps_completed,
        "stop_reason": receipt.whole_student.stop_reason,
    }


def _default_resume() -> P39SyntheticLearningReceipt:
    return run_p3_9_synthetic_learning_smoke()


def _default_observability():
    return run_p3_8_observability_acceptance()


def _default_synthetic() -> P39SyntheticLearningReceipt:
    return run_p3_9_synthetic_learning_smoke()


def _default_replay():
    return run_p3_9_synthetic_learning_smoke(), run_p3_9_synthetic_learning_smoke()


def _default_dependencies() -> P310AcceptanceDependencies:
    return P310AcceptanceDependencies(
        _contracts,
        _optimizer,
        _default_single_step,
        _default_loop,
        _checkpoint,
        _default_resume,
        _default_observability,
        _default_synthetic,
        _default_replay,
        _default_documentation,
        _default_inventory,
    )


def _finding(
    code: str, section: str, actual: Any, expected: Any = True
) -> LearningIssue:
    return LearningIssue(
        code,
        "P3.10 learning core acceptance section failed",
        {"section": section, "check": section, "expected": expected, "actual": actual},
    )


def _audit_contracts(deps: P310AcceptanceDependencies) -> bool:
    evidence = deps.contracts_fn()
    return (
        evidence["config"]["update_scope"]["kind"] == "whole_student"
        and evidence["config"]["objective_scope"]["kind"] == "final_output"
        and evidence["batch"]["batch_id"] == "p310-batch"
        and evidence["objective_result"]["loss"] == 1.0
        and evidence["learning_json"] == canonical_learning_json(evidence["config"])
        and evidence["objective_json"]
        == canonical_objective_json(evidence["objective_request"])
    )


def _audit_optimizer(deps: P310AcceptanceDependencies) -> bool:
    evidence = deps.optimizer_fn()
    return (
        evidence["optimizer_id"] == "sgd.v1"
        and evidence["step"] == 1
        and tuple(evidence["changed"]) == ("head.bias", "trunk.weight")
        and all(value == -0.1 for value in evidence["parameters"].values())
    )


def _audit_single_step(deps: P310AcceptanceDependencies) -> bool:
    evidence = deps.single_step_fn()
    return evidence["status"] == "pass" and evidence["loss_decrease"] is True


def _audit_loop(deps: P310AcceptanceDependencies) -> bool:
    evidence = deps.loop_fn()
    return (
        evidence["status"] == "pass"
        and evidence["steps"] == 12
        and evidence["stop_reason"] == "max_steps"
    )


def _audit_checkpoint(deps: P310AcceptanceDependencies) -> bool:
    evidence = deps.checkpoint_fn()
    return (
        evidence["schema"] == CHECKPOINT_SCHEMA_VERSION == "learning_checkpoint.v2"
        and evidence["source_state"]["position"] == 1
        and evidence["source_owned"]
        and evidence["source_hashed"]
        and evidence["source_sized"]
    )


def _audit_resume(deps: P310AcceptanceDependencies) -> bool:
    receipt = deps.resume_fn()
    return receipt.status == "pass" and receipt.checkpoint_restore_valid


def _audit_observability(deps: P310AcceptanceDependencies) -> bool:
    return deps.observability_fn().status == "pass"


def _audit_synthetic(deps: P310AcceptanceDependencies) -> bool:
    receipt = deps.synthetic_fn()
    return receipt.status == "pass"


def _audit_replay(deps: P310AcceptanceDependencies) -> bool:
    first, second = deps.replay_fn()
    return (
        first.status == second.status == "pass" and first.to_json() == second.to_json()
    )


def _audit_documentation(deps: P310AcceptanceDependencies) -> bool:
    evidence = deps.documentation_fn()
    return all(evidence.values())


def _audit_inventory(deps: P310AcceptanceDependencies) -> bool:
    evidence = deps.test_inventory_fn()
    return evidence["named_tests"] >= evidence["required"] and evidence["files"] == 3


def run_p3_10_learning_core_acceptance(
    dependencies: P310AcceptanceDependencies | None = None,
) -> P310LearningCoreAcceptanceReceipt:
    deps = _default_dependencies() if dependencies is None else dependencies
    audits = (
        ("contracts_valid", _audit_contracts),
        ("optimizer_valid", _audit_optimizer),
        ("single_step_valid", _audit_single_step),
        ("loop_valid", _audit_loop),
        ("checkpoint_valid", _audit_checkpoint),
        ("resume_valid", _audit_resume),
        ("observability_valid", _audit_observability),
        ("synthetic_learning_valid", _audit_synthetic),
        ("deterministic_replay_valid", _audit_replay),
        ("documentation_valid", _audit_documentation),
        ("test_inventory_valid", _audit_inventory),
    )
    values: dict[str, bool] = {}
    blockers: list[LearningIssue] = []
    for field_name, audit in audits:
        try:
            values[field_name] = bool(audit(deps))
        except Exception as exc:
            values[field_name] = False
            blockers.append(
                _finding(
                    "p3_10_internal_error",
                    field_name,
                    type(exc).__name__,
                    "no exception",
                )
            )
        if not values[field_name]:
            blockers.append(_finding(SECTION_CODES[field_name], field_name, False))
    return P310LearningCoreAcceptanceReceipt(
        schema_version=SCHEMA,
        status="pass" if all(values.values()) and not blockers else "fail",
        blockers=tuple(blockers),
        metadata={"gate": "P3.10", "section_count": len(audits)},
        **values,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_p3_10_learning_core_acceptance()
    if args.json:
        print(receipt.to_json())
    else:
        print("P3.10 Learning Core Golden Acceptance")
        print(f"  status: {receipt.status.upper()}")
        section_count = sum(getattr(receipt, name) for name in VALIDITY_FIELDS)
        print(f"  sections: {section_count}/{len(VALIDITY_FIELDS)}")
        print(f"  blockers: {len(receipt.blockers)}")
        print("  note: contracts and synthetic systems evidence, not model quality")
    return 0 if receipt.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
