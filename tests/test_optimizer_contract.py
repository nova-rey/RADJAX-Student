from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.learning import UpdateScope
from radjax_student.optimizers import (
    GradientTree,
    OptimizerBackend,
    OptimizerConfig,
    OptimizerContractError,
    OptimizerInitRequest,
    OptimizerRegistry,
    OptimizerUpdateRequest,
    SgdOptimizer,
    canonical_optimizer_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _setup(scope: UpdateScope | None = None):
    architecture = FakeArchitecturePlugin()
    catalog = architecture.describe_parameters()
    selection = architecture.resolve_update_scope(scope or UpdateScope(), catalog)
    optimizer = SgdOptimizer()
    config = OptimizerConfig(optimizer_id=optimizer.optimizer_id, learning_rate=0.5)
    state = optimizer.initialize_state(
        OptimizerInitRequest(config, catalog, selection)
    ).optimizer_state
    parameters = {"head.weight": 3.0, "trunk.bias": 2.0, "trunk.weight": 1.0}
    gradients = GradientTree(
        catalog.paths,
        values={"head.weight": 1.0, "trunk.bias": 2.0, "trunk.weight": 3.0},
    )
    return (
        architecture,
        catalog,
        selection,
        optimizer,
        config,
        state,
        parameters,
        gradients,
    )


def test_sgd_satisfies_protocol_and_has_deterministic_capabilities() -> None:
    optimizer = SgdOptimizer()
    assert isinstance(optimizer, OptimizerBackend)
    assert optimizer.capability_profile().capabilities == tuple(
        sorted(optimizer.capability_profile().capabilities)
    )


def test_config_serializes_clipping_and_weight_decay_policy() -> None:
    config = OptimizerConfig(
        optimizer_id="sgd.v1",
        learning_rate=0.25,
        gradient_clip_mode="global_norm",
        gradient_clip=1.0,
        weight_decay=0.01,
        weight_decay_mode="decoupled",
    )
    assert OptimizerConfig.from_dict(config.to_dict()) == config
    assert (
        json.loads(canonical_optimizer_json(config.to_dict()))["weight_decay_mode"]
        == "decoupled"
    )
    with pytest.raises(ValueError, match="requires gradient_clip"):
        OptimizerConfig(optimizer_id="sgd.v1", gradient_clip_mode="value")


def test_registry_rejects_duplicates_and_lists_deterministically() -> None:
    registry = OptimizerRegistry()
    optimizer = SgdOptimizer()
    registry.register(optimizer)
    assert registry.list_optimizers() == ("sgd.v1",)
    assert registry.get("sgd.v1") is optimizer
    with pytest.raises(OptimizerContractError, match="already registered"):
        registry.register(optimizer)


def test_whole_student_sgd_update_changes_all_selected_parameters() -> None:
    _, catalog, selection, optimizer, config, state, parameters, gradients = _setup()
    result = optimizer.apply_updates(
        OptimizerUpdateRequest(
            gradients, state, config, selection, 0, parameters=parameters
        )
    )
    assert result.changed_parameter_paths == catalog.paths
    assert result.updated_parameters == {
        "head.weight": 2.5,
        "trunk.bias": 1.0,
        "trunk.weight": -0.5,
    }
    assert result.updated_optimizer_state.step == 1
    assert optimizer.describe_state(result.updated_optimizer_state).state_count == 3


def test_partial_update_preserves_excluded_parameter_and_state_values() -> None:
    _, _, selection, optimizer, config, state, parameters, gradients = _setup(
        UpdateScope(kind="named_region", region_id="trunk")
    )
    result = optimizer.apply_updates(
        OptimizerUpdateRequest(
            gradients, state, config, selection, 4, parameters=parameters
        )
    )
    old_steps = state.backend_state["per_parameter_steps"]
    new_steps = result.updated_optimizer_state.backend_state["per_parameter_steps"]
    assert result.updated_parameters["head.weight"] == parameters["head.weight"]
    assert "head.weight" in result.unchanged_parameter_paths
    assert new_steps["head.weight"] == old_steps["head.weight"] == 0
    assert new_steps["trunk.weight"] == 1


def test_sgd_rejects_structure_unknown_paths_and_nonfinite_gradients() -> None:
    _, _, selection, optimizer, config, state, parameters, gradients = _setup()
    bad_gradients = GradientTree(
        state.parameter_paths,
        values={"head.weight": 1.0, "trunk.bias": 2.0, "trunk.weight": math.inf},
    )
    with pytest.raises(OptimizerContractError, match="gradient must be finite"):
        optimizer.apply_updates(
            OptimizerUpdateRequest(
                bad_gradients, state, config, selection, 0, parameters=parameters
            )
        )
    with pytest.raises(OptimizerContractError, match="must match optimizer state"):
        optimizer.apply_updates(
            OptimizerUpdateRequest(
                gradients, state, config, selection, 0, parameters={"head.weight": 1.0}
            )
        )


def test_optimizer_import_has_no_ml_runtime_architecture_execution_or_training() -> (
    None
):
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
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry
assert OptimizerConfig(optimizer_id="import-test").optimizer_id == "import-test"
assert OptimizerRegistry().list_optimizers() == ()
assert not any(name.startswith("radjax_student.runtime") for name in sys.modules)
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


def test_optimizer_source_has_no_forbidden_dependencies_or_loop_entrypoint() -> None:
    root = REPO_ROOT / "src" / "radjax_student" / "optimizers"
    forbidden = (
        "jax",
        "jaxlib",
        "flax",
        "equinox",
        "optax",
        "torch",
        "transformers",
        "datasets",
        "radjax_tome",
        "radjax_student.runtime",
        "radjax_student.training",
    )
    offenders = [
        f"{path.name} imports {name}"
        for path in root.glob("*.py")
        for name in forbidden
        if f"import {name}" in path.read_text(encoding="utf-8")
        or f"from {name}" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
