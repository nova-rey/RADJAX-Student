"""Adversarial P4.3 contracts for RWKV-7 reference initialization only."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from radjax_student.architecture import (
    ArchitectureContractError,
    ArchitectureInitRequest,
    JaxArchitecturePlugin,
)
from radjax_student.architecture.rwkv7_reference.config import (
    reference_architecture_config,
)
from radjax_student.architecture.rwkv7_reference.plugin import RWKV7ReferencePlugin
from radjax_student.architecture.rwkv7_reference.schema import (
    CARRY_PYTREE_DESCRIPTOR_DIGEST,
    carry_descriptor,
)
from radjax_student.runtime.jax_bridge import (
    RuntimeJaxBridgeError,
    materialize_initialization_jax_key,
)

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SOURCE = (
    ROOT / "src" / "radjax_student" / "architecture" / "rwkv7_reference" / "plugin.py"
)
_MISSING = object()


def _request(
    reference: str,
    *,
    precision_policy: str = "float32",
    initialization_material: object = _MISSING,
) -> ArchitectureInitRequest:
    if initialization_material is _MISSING:
        initialization_material = materialize_initialization_jax_key(reference)
    return ArchitectureInitRequest(
        config=reference_architecture_config(),
        runtime_keys_reference=reference,
        precision_policy=precision_policy,
        runtime_initialization_material=initialization_material,
    )


def _leaf_at(tree: Mapping[str, object], keypath: tuple[str, ...]) -> object:
    node: object = tree
    for key in keypath:
        assert isinstance(node, Mapping), f"{keypath} is not a mapping tree"
        node = node[key]
    return node


def _initialized(reference: str, *, initialization_material: object = _MISSING):
    return RWKV7ReferencePlugin().initialize_parameters(
        _request(reference, initialization_material=initialization_material)
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {name: _json_value(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def test_equal_runtime_references_reproduce_complete_initialization_evidence() -> None:
    jax = pytest.importorskip("jax")

    first = _initialized("runtime_keys.v1:initialization:17")
    repeated = _initialized("runtime_keys.v1:initialization:17")

    # Result metadata is separate from numerical leaves, so assert both.
    assert first.parameter_catalog == repeated.parameter_catalog
    assert first.architecture_state == repeated.architecture_state
    assert first.architecture_carry_descriptor == repeated.architecture_carry_descriptor
    assert first.parameter_layout == repeated.parameter_layout
    assert first.hf_descriptor == repeated.hf_descriptor
    assert first.hf_reference == repeated.hf_reference
    assert first.claims_not_made == repeated.claims_not_made

    assert first.parameters is not None
    assert repeated.parameters is not None
    assert first.architecture_carry is not None
    assert repeated.architecture_carry is not None
    for entry in first.parameter_layout.entries:
        assert jax.numpy.array_equal(
            _leaf_at(first.parameters, entry.jax_keypath),
            _leaf_at(repeated.parameters, entry.jax_keypath),
        ), entry.logical_path
    for name in first.architecture_carry:
        assert jax.numpy.array_equal(
            first.architecture_carry[name], repeated.architecture_carry[name]
        ), name


def test_changed_runtime_reference_changes_the_initialized_parameter_values() -> None:
    jax = pytest.importorskip("jax")

    first = _initialized("runtime_keys.v1:initialization:17")
    changed = _initialized("runtime_keys.v1:initialization:18")

    assert first.parameter_layout == changed.parameter_layout
    assert first.parameters is not None
    assert changed.parameters is not None
    assert any(
        not jax.numpy.array_equal(
            _leaf_at(first.parameters, entry.jax_keypath),
            _leaf_at(changed.parameters, entry.jax_keypath),
        )
        for entry in first.parameter_layout.entries
    )


def test_initialized_tree_carry_and_hf_reference_match_declared_contracts() -> None:
    jax = pytest.importorskip("jax")

    result = _initialized("runtime_keys.v1:initialization:19")

    assert result.parameters is not None
    result.parameter_layout.validate_materialized_parameters(result.parameters)
    observed_paths = set()
    for entry in result.parameter_layout.entries:
        leaf = _leaf_at(result.parameters, entry.jax_keypath)
        observed_paths.add(entry.logical_path)
        assert tuple(leaf.shape) == entry.shape
        assert str(leaf.dtype) == "float32"
        assert str(leaf.dtype) == entry.dtype
    assert observed_paths == set(result.parameter_catalog.paths)
    assert observed_paths == set(result.parameter_layout.logical_paths)

    declared_carry = carry_descriptor()["persistent_leaves"]
    assert isinstance(declared_carry, Mapping)
    assert _json_value(result.architecture_carry_descriptor) == {
        "schema_version": "architecture_carry.v1",
        "state_id": "rwkv7_reference_state.v1",
        "pytree_descriptor_digest": CARRY_PYTREE_DESCRIPTOR_DIGEST,
    }
    assert result.architecture_carry is not None
    assert set(result.architecture_carry) == set(declared_carry)
    for name, specification in declared_carry.items():
        value = result.architecture_carry[name]
        assert tuple(value.shape) == tuple(specification["shape"])
        assert str(value.dtype) == specification["dtype"]
        assert int(jax.numpy.count_nonzero(value)) == 0

    assert result.hf_descriptor is not None
    assert result.hf_reference == result.hf_descriptor.preservation_reference()
    assert result.hf_reference is not None
    assert (
        result.hf_reference.parameter_layout_digest == result.parameter_layout.digest()
    )


def test_initialization_rejects_non_float32_precision_with_jax_execution() -> None:
    plugin = RWKV7ReferencePlugin()

    assert isinstance(plugin, JaxArchitecturePlugin)
    assert plugin.capability_profile().supports("architecture.jax_execution_v1")
    with pytest.raises(ArchitectureContractError) as caught:
        plugin.initialize_parameters(
            _request("runtime_keys.v1:initialization:17", precision_policy="bfloat16")
        )
    assert caught.value.code == "architecture_initialization_failed"


@pytest.mark.parametrize(
    "reference",
    (
        "runtime_keys.v1:model_initialization:17",
        "runtime_keys.v1:initialization:017",
        "runtime_keys.v1:initialization:-1",
        "runtime_keys.v1:initialization:17:extra",
    ),
)
def test_malformed_initialization_references_fail_at_runtime_owner(
    reference: str,
) -> None:
    with pytest.raises(RuntimeJaxBridgeError) as direct:
        materialize_initialization_jax_key(reference)
    assert direct.value.code == "runtime_jax_initialization_reference_invalid"


def test_plugin_uses_only_runtime_supplied_initialization_material() -> None:
    jax = pytest.importorskip("jax")
    supplied = materialize_initialization_jax_key("runtime_keys.v1:initialization:17")
    first = _initialized(
        "runtime_keys.v1:initialization:17", initialization_material=supplied
    )
    same_material_different_reference = _initialized(
        "runtime_keys.v1:initialization:18", initialization_material=supplied
    )
    assert first.parameters is not None
    assert same_material_different_reference.parameters is not None
    for entry in first.parameter_layout.entries:
        assert jax.numpy.array_equal(
            _leaf_at(first.parameters, entry.jax_keypath),
            _leaf_at(same_material_different_reference.parameters, entry.jax_keypath),
        )

    with pytest.raises(ArchitectureContractError) as plugin_failure:
        RWKV7ReferencePlugin().initialize_parameters(
            _request("runtime_keys.v1:initialization:17", initialization_material=None)
        )
    assert plugin_failure.value.code == "architecture_initialization_failed"


def test_initialization_material_is_never_serialized() -> None:
    request = _request("runtime_keys.v1:initialization:17")
    payload = request.to_dict()

    assert "runtime_initialization_material" not in payload
    assert (
        ArchitectureInitRequest.from_dict(payload).runtime_initialization_material
        is None
    )


def test_canonical_initialization_and_package_import_are_pure() -> None:
    tree = ast.parse(PLUGIN_SOURCE.read_text())
    initialize = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "initialize_parameters"
    )
    imports = [
        alias.name
        for node in ast.walk(initialize)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imports.extend(
        node.module or ""
        for node in ast.walk(initialize)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imports)
    assert not any(
        name == "radjax_student.runtime" or name.startswith("radjax_student.runtime.")
        for name in imports
    )

    forbidden_host_methods = {"device_get", "tolist", "item", "to_py"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in forbidden_host_methods
        for node in ast.walk(initialize)
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import radjax_student.architecture.rwkv7_reference; "
                "assert 'jax' not in sys.modules; assert 'jaxlib' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
