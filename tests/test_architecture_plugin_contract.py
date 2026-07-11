from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.architecture import (
    ARCHITECTURE_CLAIMS_NOT_MADE,
    ArchitectureConfig,
    ArchitectureContractError,
    ArchitectureInitRequest,
    ArchitectureInitResult,
    ArchitecturePlugin,
    ArchitectureRegistry,
    ForwardRequest,
    ForwardResult,
    ParameterCatalog,
    ParameterDescriptor,
    canonical_architecture_json,
)
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.learning import LearningBatch, ObjectiveScope, UpdateScope

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plugin_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        architecture_id="test.architecture.v1",
        vocab_size=16,
        sequence_length=8,
        model_config={"test_double": True},
    )


def _batch(*, sequence_length: int = 4) -> LearningBatch:
    return LearningBatch(
        batch_id="architecture-contract-batch",
        inputs={"token_ids": {"rank": 2, "sequence_length": sequence_length}},
    )


def test_fake_plugin_satisfies_the_architecture_protocol() -> None:
    plugin = FakeArchitecturePlugin()

    assert isinstance(plugin, ArchitecturePlugin)
    assert plugin.architecture_id == "test.architecture.v1"
    assert plugin.capability_profile().capabilities == tuple(
        sorted(plugin.capability_profile().capabilities)
    )


def test_architecture_config_and_metadata_are_deterministically_serializable() -> None:
    plugin = FakeArchitecturePlugin()
    config = _plugin_config()
    metadata = plugin.architecture_metadata()

    assert ArchitectureConfig.from_dict(config.to_dict()) == config
    encoded = canonical_architecture_json(metadata.to_dict())
    assert encoded == canonical_architecture_json(metadata.to_dict())
    assert json.loads(encoded)["parameter_count"] == 3
    assert metadata.parameter_catalog.paths == (
        "head.weight",
        "trunk.bias",
        "trunk.weight",
    )
    assert any(
        issue.code == "architecture_named_regions_overlap"
        for issue in metadata.warnings
    )


def test_parameter_catalog_requires_stable_unique_paths() -> None:
    descriptor = ParameterDescriptor(
        path="head.weight", shape=(4, 4), dtype="float32", role="output_head"
    )

    with pytest.raises(ValueError, match="unique"):
        ParameterCatalog(
            architecture_id="test.architecture.v1",
            parameters=(descriptor, descriptor),
        )
    with pytest.raises(ValueError, match="stable dotted"):
        ParameterDescriptor(path="/head/weight", shape=(1,), dtype="float32")


def test_update_scope_resolution_is_architecture_owned() -> None:
    plugin = FakeArchitecturePlugin()
    catalog = plugin.describe_parameters()

    whole = plugin.resolve_update_scope(UpdateScope(), catalog)
    region = plugin.resolve_update_scope(
        UpdateScope(kind="named_region", region_id="trunk"), catalog
    )
    explicit = plugin.resolve_update_scope(
        UpdateScope(kind="parameter_paths", parameter_paths=("head.weight",)),
        catalog,
    )

    assert whole.selected_parameter_paths == catalog.trainable_paths
    assert region.selected_parameter_paths == ("trunk.bias", "trunk.weight")
    assert explicit.selected_parameter_paths == ("head.weight",)
    with pytest.raises(ArchitectureContractError, match="unknown parameter path"):
        plugin.resolve_update_scope(
            UpdateScope(kind="parameter_paths", parameter_paths=("unknown.weight",)),
            catalog,
        )
    with pytest.raises(ArchitectureContractError, match="named update region"):
        plugin.resolve_update_scope(
            UpdateScope(kind="named_region", region_id="not-a-region"), catalog
        )


def test_objective_scope_resolution_remains_independent_of_update_scope() -> None:
    plugin = FakeArchitecturePlugin()
    metadata = plugin.architecture_metadata()
    objective = plugin.resolve_objective_scope(
        ObjectiveScope(kind="intermediate_surface", target_id="trunk_output"),
        metadata,
    )
    update = plugin.resolve_update_scope(
        UpdateScope(kind="named_region", region_id="head"),
        metadata.parameter_catalog,
    )

    assert objective.surface_id == "trunk_output"
    assert update.selected_parameter_paths == ("head.weight",)
    assert (
        plugin.resolve_objective_scope(ObjectiveScope(), metadata).surface_id
        == "final_output"
    )
    with pytest.raises(ArchitectureContractError, match="not declared"):
        plugin.resolve_objective_scope(
            ObjectiveScope(kind="intermediate_surface", target_id="missing_surface"),
            metadata,
        )


def test_batch_validation_and_passive_init_forward_models() -> None:
    plugin = FakeArchitecturePlugin()
    config = _plugin_config()

    assert plugin.validate_batch(_batch(), config).ok
    invalid = plugin.validate_batch(LearningBatch(batch_id="invalid"), config)
    assert not invalid.ok
    assert invalid.blockers[0].code == "architecture_batch_incompatible"

    initialized = plugin.initialize_parameters(
        request=ArchitectureInitRequest(
            config=config,
            runtime_keys_reference="runtime_keys.v1:model_initialization:7",
        )
    )
    forward = plugin.forward(
        ForwardRequest(batch=_batch(), parameters=initialized.parameters)
    )

    assert initialized.to_dict()["parameters_present"] is True
    assert "parameters" not in initialized.to_dict()
    assert forward.to_dict()["outputs_present"] is True
    assert "outputs" not in forward.to_dict()
    assert ArchitectureInitResult.from_dict(initialized.to_dict()) == initialized
    assert ForwardResult.from_dict(forward.to_dict()) == forward
    assert (
        ForwardRequest.from_dict(
            (forward_request := ForwardRequest(batch=_batch())).to_dict()
        )
        == forward_request
    )


def test_architecture_registry_is_explicit_and_deterministic() -> None:
    registry = ArchitectureRegistry()
    plugin = FakeArchitecturePlugin()

    registry.register(plugin)
    assert registry.get(plugin.architecture_id) is plugin
    assert registry.list_plugins() == (plugin.architecture_id,)
    with pytest.raises(ArchitectureContractError, match="already registered"):
        registry.register(plugin)
    with pytest.raises(ArchitectureContractError, match="not registered"):
        registry.get("missing")


def test_architecture_contract_imports_without_ml_or_runtime_execution() -> None:
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
from radjax_student.architecture import ArchitectureConfig, ArchitectureRegistry
config = ArchitectureConfig(architecture_id="import-test")
assert config.architecture_id == "import-test"
assert ArchitectureRegistry().list_plugins() == ()
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


def test_architecture_source_has_no_forbidden_dependencies_or_training_entrypoint() -> (
    None
):
    root = REPO_ROOT / "src" / "radjax_student" / "architecture"
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
        "radjax_student.runtime",
        "radjax_student.training",
    )
    offenders: list[str] = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden_imports:
            if f"import {name}" in text or f"from {name}" in text:
                offenders.append(f"{path.name} imports {name}")

    assert offenders == []
    assert "training_loop_not_run" in ARCHITECTURE_CLAIMS_NOT_MADE
