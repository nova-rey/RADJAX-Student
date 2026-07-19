"""Adversarial contract tests for the static P4.2 RWKV-7 reference plugin."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

from radjax_student.architecture import (
    ArchitectureConfig,
    ArchitectureContractError,
    ArchitectureInitResult,
    ArchitecturePlugin,
    ArchitectureRegistry,
    JaxArchitecturePlugin,
)
from radjax_student.architecture.rwkv7_reference import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    RWKV7ReferenceConfig,
    hf_descriptor,
    parameter_catalog,
    parameter_layout,
    pinned_numpy_parameter_order,
    reference_architecture_config,
    register_rwkv7_reference,
    validate_reference_config,
)
from radjax_student.architecture.rwkv7_reference.plugin import RWKV7ReferencePlugin
from radjax_student.contracts import (
    HFContractError,
    ObjectiveConfig,
    ObjectiveScope,
    ParameterTreeLayout,
    UpdateScope,
    validate_hf_descriptor_match,
)
from radjax_student.learning import LearningState
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry, SgdOptimizer
from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
RWKV_ROOT = REPO_ROOT / "src" / "radjax_student" / "architecture" / "rwkv7_reference"


def _frozen_architecture_config() -> ArchitectureConfig:
    """Hand-written P4.2 domain oracle, independent of the projection helper."""

    return ArchitectureConfig(
        architecture_id="radjax.architecture.rwkv7_reference",
        vocab_size=16,
        sequence_length=4,
        dtype_intent="float32",
        model_config={
            "ffn_width": 16,
            "head_count": 2,
            "head_size": 4,
            "hidden_size": 8,
            "layer_count": 2,
            "time_aaa_rank": 32,
            "time_decay_rank": 32,
            "time_gate_rank": 32,
            "time_value_rank": 32,
            "vocabulary_size": 16,
        },
    )


def _alternate(value: object) -> object:
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"not-{value}"
    raise AssertionError(f"no adversarial replacement for {value!r}")


def test_frozen_reference_config_rejects_every_declared_dimension_deviation() -> None:
    expected = {
        "vocabulary_size": 16,
        "hidden_size": 8,
        "layer_count": 2,
        "head_size": 4,
        "head_count": 2,
        "ffn_width": 16,
        "context_length": 4,
        "dtype": "float32",
        "time_decay_rank": 32,
        "time_aaa_rank": 32,
        "time_value_rank": 32,
        "time_gate_rank": 32,
    }
    assert {field.name for field in fields(RWKV7ReferenceConfig)} == set(expected)
    assert RWKV7ReferenceConfig() == RWKV7ReferenceConfig(**expected)

    for name, value in expected.items():
        altered = {**expected, name: _alternate(value)}
        with pytest.raises(ValueError, match="frozen in P4.2"):
            RWKV7ReferenceConfig(**altered)


def test_frozen_generic_projection_rejects_identity_and_each_dimension_drift() -> None:
    expected = _frozen_architecture_config()
    assert reference_architecture_config() == expected

    deviations = [
        replace(expected, architecture_id="foreign.architecture"),
        replace(expected, vocab_size=17),
        replace(expected, sequence_length=5),
        replace(expected, dtype_intent="bfloat16"),
        replace(expected, metadata={"unexpected": True}),
        replace(expected, model_config={**expected.model_config, "unexpected": 1}),
    ]
    deviations.extend(
        replace(
            expected,
            model_config={**expected.model_config, name: _alternate(value)},
        )
        for name, value in expected.model_config.items()
    )

    for config in deviations:
        with pytest.raises(ArchitectureContractError) as caught:
            validate_reference_config(config)
        assert caught.value.code == "architecture_config_invalid"


def test_static_plugin_registers_only_at_its_bound_identity_and_version() -> None:
    for plugin in (
        replace(RWKV7ReferencePlugin(), architecture_id="foreign.architecture"),
        replace(RWKV7ReferencePlugin(), architecture_version=2),
    ):
        with pytest.raises(ArchitectureContractError) as caught:
            ArchitectureRegistry().register(plugin)
        assert caught.value.code == "architecture_capability_missing"


def test_hf_descriptor_boundary_rejects_identity_and_version_mismatch() -> None:
    expected = hf_descriptor(_frozen_architecture_config())
    for observed in (
        replace(expected, architecture_id="foreign.architecture"),
        replace(expected, architecture_plugin_version=2),
    ):
        with pytest.raises(HFContractError) as caught:
            validate_hf_descriptor_match(expected, observed)
        assert caught.value.code == "hf_architecture_identity_mismatch"


def test_schema_rejects_mapping_layout_and_hf_projection_malformation() -> None:
    plugin = RWKV7ReferencePlugin()
    catalog = parameter_catalog()
    layout = parameter_layout()
    descriptor = hf_descriptor(_frozen_architecture_config())

    with pytest.raises(ArchitectureContractError) as caught:
        plugin.describe_parameters(parameters={"emb": {"weight": object()}})
    assert caught.value.code == "architecture_parameter_catalog_invalid"

    duplicate_keypath = replace(
        layout.entries[1], jax_keypath=layout.entries[0].jax_keypath
    )
    with pytest.raises(ValueError, match="bijective"):
        ParameterTreeLayout(
            layout.architecture_id,
            (layout.entries[0], duplicate_keypath, *layout.entries[2:]),
        )

    malformed_projection = replace(descriptor.parameter_projections[0], shape=(99,))
    descriptor_with_wrong_projection = replace(
        descriptor,
        parameter_projections=(
            malformed_projection,
            *descriptor.parameter_projections[1:],
        ),
    )
    with pytest.raises(ValueError, match="projection conflicts with parameter layout"):
        ArchitectureInitResult(
            parameter_catalog=catalog,
            parameter_layout=layout,
            hf_descriptor=descriptor_with_wrong_projection,
        )


def test_pinned_numpy_parameter_order_is_literal_complete_and_first_block_safe() -> (
    None
):
    order = pinned_numpy_parameter_order()
    assert tuple(order) == (
        "emb",
        "blocks.0.ln0",
        "blocks.0.ln1",
        "blocks.0.att",
        "blocks.0.ln2",
        "blocks.0.ffn",
        "blocks.1.ln1",
        "blocks.1.att",
        "blocks.1.ln2",
        "blocks.1.ffn",
        "ln_out",
        "head",
    )
    assert order["blocks.0.att"] == (
        "blocks.0.att.x_r",
        "blocks.0.att.x_w",
        "blocks.0.att.x_k",
        "blocks.0.att.x_v",
        "blocks.0.att.x_a",
        "blocks.0.att.x_g",
        "blocks.0.att.w0",
        "blocks.0.att.r_k",
        "blocks.0.att.w1",
        "blocks.0.att.w2",
        "blocks.0.att.a1",
        "blocks.0.att.a2",
        "blocks.0.att.a0",
        "blocks.0.att.g1",
        "blocks.0.att.g2",
        "blocks.0.att.k_k",
        "blocks.0.att.k_a",
        "blocks.0.att.receptance.weight",
        "blocks.0.att.key.weight",
        "blocks.0.att.value.weight",
        "blocks.0.att.output.weight",
        "blocks.0.att.ln_x.weight",
        "blocks.0.att.ln_x.bias",
    )
    assert order["blocks.1.att"][15:18] == (
        "blocks.1.att.v2",
        "blocks.1.att.v1",
        "blocks.1.att.v0",
    )
    consumed_paths = {path for prefix_paths in order.values() for path in prefix_paths}
    catalog = parameter_catalog()
    assert consumed_paths == set(catalog.paths)
    assert catalog.get("blocks.0.att.r_k").metadata["representation"] == (
        "pinned_numpy_flat_to_head_matrix"
    )
    assert catalog.get("blocks.1.att.v0").metadata["representation"] == (
        "pinned_numpy_squeeze_to_vector"
    )


def test_explicit_static_registration_is_caller_owned_and_deterministic() -> None:
    registry = ArchitectureRegistry()
    plugin = register_rwkv7_reference(registry)

    assert plugin.architecture_id == RWKV7_REFERENCE_ARCHITECTURE_ID
    assert plugin.architecture_version == RWKV7_REFERENCE_ARCHITECTURE_VERSION
    assert registry.get(RWKV7_REFERENCE_ARCHITECTURE_ID) is plugin
    assert registry.list_plugins() == (RWKV7_REFERENCE_ARCHITECTURE_ID,)


def test_static_plugin_is_an_architecture_plugin_but_not_a_jax_plugin() -> None:
    plugin = RWKV7ReferencePlugin()

    assert isinstance(plugin, ArchitecturePlugin)
    assert not isinstance(plugin, JaxArchitecturePlugin)
    assert not plugin.capability_profile().supports("architecture.jax_execution_v1")


def _assembly_request_and_registries(plugin: ArchitecturePlugin):
    pytest.importorskip("jax", reason="P3.12C assembly boundary requires JAX")
    from radjax_student.learning.assembly import (
        JaxLearningAssemblyRegistries,
        JaxLearningAssemblyRequest,
    )

    architecture_registry = ArchitectureRegistry()
    architecture_registry.register(plugin)
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(SgdOptimizer())
    request = JaxLearningAssemblyRequest(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        architecture_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        architecture_config=_frozen_architecture_config(),
        objective_identity=CANONICAL_MSE_IDENTITY,
        objective_config=ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"}),
        optimizer_id="sgd.v1",
        optimizer_version=1,
        optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.25),
        runtime_backend_id="jax",
        runtime_implementation_version="p2.9",
        runtime_config=RuntimeConfig(
            backend_id="jax",
            platform_preference="cpu",
            precision_policy="float32",
            placement_policy="single_device",
            compilation_policy="eager",
            distributed_policy="disabled",
            fallback_policy="disallowed",
            seed=17,
        ),
        root_seed=17,
        learning_state=LearningState(
            "rwkv7-static-rejection",
            active_update_scope=UpdateScope("whole_student"),
            active_objective_scope=ObjectiveScope(),
        ),
    )
    return request, JaxLearningAssemblyRegistries(
        architecture_registry,
        build_default_objective_registry(),
        optimizer_registry,
        build_default_runtime_registry(),
    )


def test_executable_assembly_rejects_static_rwkv_before_execution() -> None:
    pytest.importorskip("jax", reason="P3.12C assembly boundary requires JAX")
    from radjax_student.learning.assembly import (
        LearningAssemblyError,
        assemble_jax_learning_lifecycle,
    )

    calls = {"initializer": 0, "forward": 0}

    class ProbeStaticRWKV7(RWKV7ReferencePlugin):
        def initialize_parameters(self, request):
            del request
            calls["initializer"] += 1
            raise AssertionError("assembly called the unavailable initializer")

        def forward(self, request):
            del request
            calls["forward"] += 1
            raise AssertionError("assembly called the unavailable forward")

    request, registries = _assembly_request_and_registries(ProbeStaticRWKV7())
    with pytest.raises(LearningAssemblyError) as caught:
        assemble_jax_learning_lifecycle(request, registries=registries)

    assert caught.value.code == "assembly_architecture_invalid"
    assert calls == {"initializer": 0, "forward": 0}


def test_rwkv_subpackage_keeps_base_imports_jax_and_owner_free() -> None:
    forbidden_owner_prefixes = (
        "radjax_student.learning",
        "radjax_student.runtime",
        "radjax_student.optimizers",
        "radjax_student.validation",
    )
    offenders: list[str] = []
    for path in sorted(RWKV_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            else:
                continue
            for name in names:
                if name in forbidden_owner_prefixes or name.startswith(
                    tuple(f"{prefix}." for prefix in forbidden_owner_prefixes)
                ):
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")

        for node in tree.body:
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            else:
                continue
            for name in names:
                if name == "jax" or name.startswith("jax."):
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")

    assert offenders == []

    script = """
import builtins
import sys
real_import = builtins.__import__
forbidden = {
    'radjax_student.learning', 'radjax_student.runtime',
    'radjax_student.optimizers', 'radjax_student.validation', 'jax', 'jaxlib',
}
def guarded(name, *args, **kwargs):
    if name in forbidden or name.startswith(tuple(item + '.' for item in forbidden)):
        raise AssertionError(f'forbidden import: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import radjax_student.architecture.rwkv7_reference
assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules
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
