"""M2.6 proof that generic v3 persistence preserves Mamba-2 state."""

from __future__ import annotations

# JAX availability is checked before importing JAX-bearing production modules.
# ruff: noqa: E402, I001
import json
from dataclasses import replace
from pathlib import Path

import pytest

jax = pytest.importorskip("jax", reason="Mamba-2 checkpoint tests require JAX")
jnp = pytest.importorskip("jax.numpy", reason="Mamba-2 checkpoint tests require JAX")

from radjax_student.checkpoints import (
    CheckpointValidationError,
    load_learning_checkpoint_v3,
    save_learning_checkpoint_v3,
)
from radjax_student.learning.jax_batch import FiniteJsonJaxBatchMaterializer
from tests.support.mamba2_learning import assembled, batch, execute, tree_allclose

pytestmark = pytest.mark.jax


def _save_after_real_step(tmp_path: Path):
    source = assembled()
    first = execute(source, batch())
    lifecycle = source.loop_executor.lifecycle
    directory = tmp_path / "mamba2-v3"
    saved = save_learning_checkpoint_v3(
        lifecycle.checkpoint(), directory, optimizer=lifecycle.optimizer
    )
    return source, first, lifecycle, directory, saved


def _owner_bound_load(directory: Path, lifecycle, **overrides):
    expected = {
        "optimizer": lifecycle.optimizer,
        "parameter_layout": lifecycle.parameter_layout,
        "runtime_reference": lifecycle.runtime_reference,
        "expected_hf_reference": lifecycle.hf_reference,
        "expected_hf_descriptor": lifecycle.hf_descriptor,
        "expected_architecture_config_digest": lifecycle.config_digest,
        "expected_parameter_catalog_digest": lifecycle.catalog_digest,
        "expected_architecture_state_id": lifecycle.architecture_state.state_id,
        "expected_architecture_carry_descriptor": (
            lifecycle.architecture_carry_descriptor
        ),
        "expected_objective_descriptor": lifecycle.objective_descriptor,
        "expected_objective_config": lifecycle.objective_config,
        "expected_resolved_objective_selection": lifecycle.resolved_objective_selection,
        "expected_objective_selection": lifecycle.objective_selection,
    }
    expected.update(overrides)
    return load_learning_checkpoint_v3(directory, **expected)


def _forward(lifecycle):
    materialized = FiniteJsonJaxBatchMaterializer().materialize(batch())
    return lifecycle.architecture.apply_jax(
        jax.tree_util.tree_map(jnp.asarray, lifecycle.parameters),
        jax.tree_util.tree_map(jnp.asarray, lifecycle.architecture_carry),
        materialized,
        architecture_config=lifecycle.architecture_config,
        objective_scope=lifecycle.learning_state.active_objective_scope,
        training=False,
        rng_key=None,
    )


def test_generic_checkpoint_restores_mamba_state_and_replays(tmp_path) -> None:
    source, first, after, directory, saved = _save_after_real_step(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    descriptor = json.loads((directory / "hf_descriptor.json").read_text())
    assert first.result.status == "pass"
    assert (
        manifest["architecture"]["architecture_id"]
        == after.architecture.architecture_id
    )
    assert (
        manifest["architecture"]["parameter_layout_digest"]
        == after.parameter_layout.digest()
    )
    assert manifest["architecture"]["architecture_carry_descriptor"] == dict(
        after.architecture_carry_descriptor
    )
    assert (
        descriptor["architecture_plugin_version"]
        == after.architecture.architecture_version
    )
    assert saved.architecture_config_digest == after.config_digest
    assert saved.parameter_catalog_digest == after.catalog_digest
    assert saved.architecture_carry_descriptor == after.architecture_carry_descriptor

    restored_assembly = assembled()
    restored = restored_assembly.lifecycle.restore_from_checkpoint(directory)
    assert restored.architecture_config == after.architecture_config
    assert restored.parameter_layout == after.parameter_layout
    assert restored.architecture_state == after.architecture_state
    assert restored.architecture_carry_descriptor == after.architecture_carry_descriptor
    assert restored.hf_descriptor == after.hf_descriptor
    assert restored.hf_reference == after.hf_reference
    assert restored.learning_state == after.learning_state
    assert restored.optimizer_state.envelope == after.optimizer_state.envelope
    assert tree_allclose(restored.parameters, after.parameters)
    assert tree_allclose(restored.architecture_carry, after.architecture_carry)
    assert tree_allclose(restored.optimizer_state.arrays, after.optimizer_state.arrays)
    assert tree_allclose(_forward(after).outputs, _forward(restored).outputs)
    assert tree_allclose(
        _forward(after).updated_architecture_carry,
        _forward(restored).updated_architecture_carry,
    )

    restored_assembly.loop_executor.lifecycle = restored
    source_next = execute(source, batch((5, 3, 7, 1)))
    restored_next = execute(restored_assembly, batch((5, 3, 7, 1)))
    assert source_next.result.loss.loss == pytest.approx(restored_next.result.loss.loss)
    assert (
        source_next.runtime_result.callable_reference
        == restored_next.runtime_result.callable_reference
    )
    assert tree_allclose(
        source.loop_executor.lifecycle.parameters,
        restored_assembly.loop_executor.lifecycle.parameters,
    )
    assert tree_allclose(
        source.loop_executor.lifecycle.architecture_carry,
        restored_assembly.loop_executor.lifecycle.architecture_carry,
    )


def test_checkpoint_restore_rejects_config_layout_state_and_hf_mismatch(
    tmp_path,
) -> None:
    _, _, lifecycle, directory, _ = _save_after_real_step(tmp_path)
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_config_identity_mismatch"
    ):
        _owner_bound_load(
            directory, lifecycle, expected_architecture_config_digest="foreign-config"
        )
    with pytest.raises(CheckpointValidationError, match="checkpoint_layout_mismatch"):
        _owner_bound_load(
            directory,
            lifecycle,
            parameter_layout=replace(
                lifecycle.parameter_layout,
                architecture_id="radjax.architecture.foreign",
            ),
        )
    with pytest.raises(
        CheckpointValidationError,
        match="checkpoint_architecture_state_identity_mismatch",
    ):
        _owner_bound_load(
            directory,
            lifecycle,
            expected_architecture_state_id="mamba2_reference_state.foreign",
        )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_architecture_descriptor_mismatch"
    ):
        _owner_bound_load(
            directory,
            lifecycle,
            expected_architecture_carry_descriptor={
                **dict(lifecycle.architecture_carry_descriptor),
                "pytree_descriptor_digest": "f" * 64,
            },
        )
    foreign_descriptor = replace(
        lifecycle.hf_descriptor,
        tokenizer=replace(
            lifecycle.hf_descriptor.tokenizer,
            tokenizer_id="mamba2-reference-foreign-tokenizer",
        ),
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_hf_identity_mismatch"
    ):
        _owner_bound_load(
            directory,
            lifecycle,
            expected_hf_reference=foreign_descriptor.preservation_reference(),
            expected_hf_descriptor=foreign_descriptor,
        )
