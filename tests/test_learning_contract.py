from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.learning import (
    LEARNING_CLAIMS_NOT_MADE,
    CheckpointPolicy,
    LearningConfig,
    LearningIssue,
    LearningReport,
    LearningState,
    LearningStepResult,
    LossResult,
    MetricRecord,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
    canonical_learning_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_scopes_are_boring_and_independent() -> None:
    config = LearningConfig()

    assert config.update_scope == UpdateScope(kind="whole_student")
    assert config.objective_scope == ObjectiveScope(kind="final_output")
    assert UpdateScope(kind="named_region", region_id="adapter") != ObjectiveScope(
        kind="named_region", target_id="adapter"
    )


def test_named_scopes_round_trip_without_interpreting_identifiers() -> None:
    update_scope = UpdateScope(
        kind="named_region",
        region_id="middle.adapter",
        metadata={"owner": "architecture-plugin"},
    )
    objective_scope = ObjectiveScope(
        kind="intermediate_surface",
        target_id="hidden.after_block_6",
    )

    assert UpdateScope.from_dict(update_scope.to_dict()) == update_scope
    assert ObjectiveScope.from_dict(objective_scope.to_dict()) == objective_scope
    assert update_scope.region_id == "middle.adapter"
    assert objective_scope.target_id == "hidden.after_block_6"


def test_parameter_paths_are_stable_unique_and_disjoint_from_exclusions() -> None:
    scope = UpdateScope(
        kind="parameter_paths",
        parameter_paths=("blocks/0/adapter", "head/weight"),
    )
    selection = ResolvedUpdateSelection(
        selection_id="architecture-selection-1",
        selected_parameter_paths=scope.parameter_paths,
        excluded_parameter_paths=("head/bias",),
    )

    assert UpdateScope.from_dict(scope.to_dict()) == scope
    assert ResolvedUpdateSelection.from_dict(selection.to_dict()) == selection
    with pytest.raises(ValueError, match="duplicates"):
        UpdateScope(kind="parameter_paths", parameter_paths=("head/weight",) * 2)
    with pytest.raises(Exception, match="stable relative"):
        UpdateScope(kind="parameter_paths", parameter_paths=("/head/weight",))
    with pytest.raises(Exception, match="overlap"):
        ResolvedUpdateSelection(
            selection_id="bad-selection",
            selected_parameter_paths=("head/weight",),
            excluded_parameter_paths=("head/weight",),
        )


def test_invalid_scope_identifiers_are_rejected() -> None:
    with pytest.raises(Exception, match="requires region_id"):
        UpdateScope(kind="named_region")
    with pytest.raises(ValueError, match="nonempty"):
        ObjectiveScope(kind="intermediate_surface", target_id="")
    with pytest.raises(Exception, match="unsupported"):
        UpdateScope(kind="unknown")  # type: ignore[arg-type]


def test_learning_models_round_trip_through_deterministic_json() -> None:
    metric = MetricRecord(name="loss", value=0.25, step=3, aggregation="mean")
    loss = LossResult(
        loss=0.25,
        components={"primary": 0.25},
        metrics=(metric,),
        objective_scope=ObjectiveScope(kind="final_output"),
    )
    step = LearningStepResult(
        status="pass",
        global_step_before=2,
        global_step_after=3,
        loss=loss,
        metrics=(metric,),
        active_update_scope=UpdateScope(
            kind="parameter_paths", parameter_paths=("head/weight",)
        ),
    )
    report = LearningReport(
        status="pass",
        config=LearningConfig(seed_reference=7),
        state=LearningState(run_id="learning-contract-test", global_step=3),
        latest_step=step,
        metrics=(metric,),
    )

    payload = report.to_dict()
    encoded = canonical_learning_json(payload)

    assert LearningReport.from_dict(json.loads(encoded)) == report
    assert encoded == canonical_learning_json(payload)
    assert encoded.endswith(b"\n")
    assert "optimizer_state" not in payload["state"]
    assert "parameter_tree" not in payload


def test_learning_state_and_checkpoint_policy_validate_transitions() -> None:
    with pytest.raises(ValueError, match="global_step"):
        LearningState(run_id="bad", global_step=-1)
    with pytest.raises(ValueError, match="every_n_steps"):
        CheckpointPolicy(mode="every_n_steps")
    with pytest.raises(ValueError, match="monitor_metric"):
        CheckpointPolicy(mode="on_improvement")
    with pytest.raises(ValueError, match="cannot precede"):
        LearningStepResult(status="pass", global_step_before=4, global_step_after=3)


def test_learning_issues_are_structured_and_serializable() -> None:
    issue = LearningIssue(
        code="learning_scope_capability_missing",
        message="The architecture cannot resolve the requested scope.",
        details={"capability": "adapter_scope"},
    )

    assert LearningIssue.from_dict(issue.to_dict()) == issue
    assert issue.to_dict()["details"] == {"capability": "adapter_scope"}


def test_learning_contract_import_does_not_load_ml_or_architecture_stacks() -> None:
    script = """
import builtins
import sys
real_import = builtins.__import__
forbidden = {
    "jax", "jaxlib", "flax", "equinox", "optax", "torch", "transformers",
    "datasets", "radjax_tome",
}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from radjax_student.learning import LearningConfig, LearningState
assert LearningConfig().update_scope.kind == "whole_student"
assert LearningState(run_id="isolated").global_step == 0
assert not any(name.startswith("radjax_student.architecture") for name in sys.modules)
assert not any(name.startswith("radjax_student.training") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_learning_source_has_no_execution_or_forbidden_dependencies() -> None:
    root = REPO_ROOT / "src" / "radjax_student" / "learning"
    forbidden_imports = (
        "jax",
        "jaxlib",
        "flax",
        "equinox",
        "optax",
        "torch",
        "transformers",
        "datasets",
        "radjax_tome",
        "radjax_student.architecture",
        "radjax_student.training",
    )
    forbidden_public_names = ("train", "execute", "optimize")
    offenders: list[str] = []
    for path in root.glob("*.py"):
        if path.name in {"jax_core.py", "jax_execution.py", "p3_5_acceptance.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for name in forbidden_imports:
            if f"import {name}" in text or f"from {name}" in text:
                offenders.append(f"{path.name} imports {name}")
    package = __import__("radjax_student.learning", fromlist=["*"])
    public_names = tuple(name for name in dir(package) if not name.startswith("_"))

    assert offenders == []
    assert not any(name.startswith(forbidden_public_names) for name in public_names)
    assert "gradient_not_computed" in LEARNING_CLAIMS_NOT_MADE
