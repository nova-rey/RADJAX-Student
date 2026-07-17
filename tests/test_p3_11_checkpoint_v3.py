"""P3.11.7 proves optimizer-owned step identity in checkpoint v3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from radjax_student.architecture import ArchitectureState
from radjax_student.checkpoints import (
    CheckpointValidationError,
    JaxLearningCheckpointV3,
    inspect_historical_v3_objective_alias,
    save_learning_checkpoint_v3,
)
from radjax_student.checkpoints import (
    load_learning_checkpoint_v3 as _load_learning_checkpoint_v3,
)
from radjax_student.checkpoints.npz_codec import (
    descriptor_digest,
    read_deterministic_npz,
    write_deterministic_npz,
)
from radjax_student.contracts import (
    HFArchitectureProjection,
    HFCompatibilityDescriptor,
    HFParameterProjection,
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    JaxOptimizerStateDescriptor,
    ObjectiveConfig,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
    ResolvedObjectiveSelection,
)
from radjax_student.learning import LearningState, ObjectiveScope
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import (
    JaxOptimizerState,
    OptimizerCapabilityProfile,
    OptimizerConfig,
    OptimizerState,
    SgdOptimizer,
    advanced_jax_optimizer_state,
    validate_jax_optimizer_state,
)


def _layout() -> ParameterTreeLayout:
    return ParameterTreeLayout(
        "test.checkpoint.architecture.v1",
        (ParameterTreeLayoutEntry("w", ("w",), (1,), "float32", "other"),),
    )


def _state(optimizer: SgdOptimizer, step: int = 2) -> JaxOptimizerState:
    layout = _layout()
    envelope = OptimizerState(optimizer.optimizer_id, layout.logical_paths, step=step)
    descriptor = optimizer.jax_state_descriptor(layout)
    return JaxOptimizerState(
        envelope,
        descriptor,
        {
            "step": np.asarray(step, dtype=np.int32),
            "per_parameter_steps": {"w": np.asarray(step, dtype=np.int32)},
        },
    )


def _objective_identity():
    registry = build_default_objective_registry()
    selection = registry.select(CANONICAL_MSE_IDENTITY)
    config = ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"})
    resolved = ResolvedObjectiveSelection(ObjectiveScope(), "final_output")
    descriptor = registry.execution_descriptor(
        selection=selection,
        config=config,
        resolved_selection=resolved,
    )
    return selection, config, resolved, descriptor


def load_learning_checkpoint_v3(directory, *, optimizer, parameter_layout, **kwargs):
    """Test fixture always models caller-bound P3.12 continuation restore."""

    selection, config, resolved, descriptor = _objective_identity()
    expected = _checkpoint(_state(optimizer))
    kwargs.setdefault("expected_hf_reference", expected.hf_reference)
    kwargs.setdefault("expected_hf_descriptor", expected.hf_descriptor)
    return _load_learning_checkpoint_v3(
        directory,
        optimizer=optimizer,
        parameter_layout=parameter_layout,
        expected_objective_descriptor=descriptor,
        expected_objective_config=config,
        expected_resolved_objective_selection=resolved,
        expected_objective_selection=selection,
        **kwargs,
    )


def _checkpoint(state: JaxOptimizerState) -> JaxLearningCheckpointV3:
    layout = _layout()
    config_digest = _digest({"architecture_config": "test"})
    objective_selection, objective_config, resolved_selection, objective_descriptor = (
        _objective_identity()
    )
    return JaxLearningCheckpointV3(
        "runtime-1",
        LearningState(
            "run", global_step=state.envelope.step, optimizer_step=state.envelope.step
        ),
        state,
        {"w": np.asarray((1.0,), dtype=np.float32)},
        {"count": np.asarray(state.envelope.step, dtype=np.int32)},
        layout,
        ArchitectureState("architecture-state-1"),
        _hf_descriptor(layout, config_digest),
        _hf_descriptor(layout, config_digest).preservation_reference(),
        config_digest,
        _digest({"parameter_catalog": "test"}),
        objective_config=objective_config,
        resolved_objective_selection=resolved_selection,
        objective_descriptor=objective_descriptor,
        objective_registry_selection=objective_selection.to_dict(),
    )


def _digest(value) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _hf_descriptor(layout, config_digest):
    return HFCompatibilityDescriptor(
        schema_version="hf_compatibility_descriptor.v2",
        architecture_id=layout.architecture_id,
        architecture_plugin_version=1,
        model_type="checkpoint_test_model",
        architecture_config_digest=config_digest,
        parameter_catalog_digest=_digest({"parameter_catalog": "test"}),
        parameter_layout_digest=layout.digest(),
        tokenizer=HFTokenizerIdentity(
            "tokenizer-test",
            "synthetic-r1",
            _digest({"tokenizer": "test"}),
            _digest({"config": "test"}),
            "synthetic",
            _digest({"norm": "identity"}),
            "synthetic",
        ),
        vocabulary=HFVocabularyIdentity(
            8,
            _digest({"vocabulary": 8}),
            _digest({"mapping": 8}),
            _digest({"added": []}),
            "0-3",
        ),
        special_tokens=HFSpecialTokenIdentity(0, 1, 2, 3, None),
        parameter_projections=tuple(
            HFParameterProjection(
                entry.logical_path,
                entry.jax_keypath,
                entry.shape,
                entry.dtype,
                "non_exportable",
                None,
                "identity",
                entry.tied_weight_group,
                "checkpoint_test_not_exported",
            )
            for entry in layout.entries
        ),
        architecture_projection=HFArchitectureProjection(
            "checkpoint_test_config", "checkpoint_test_architecture", 1, 1, 8, 1, {}
        ),
        non_claims=("no_hf_export",),
        notes="Checkpoint test fixture.",
    )


def _save(tmp_path, state=None):
    optimizer = SgdOptimizer()
    save_learning_checkpoint_v3(
        _checkpoint(state or _state(optimizer)), tmp_path, optimizer=optimizer
    )
    return optimizer


def _carry_identity(tmp_path, *, digest=None, schema="architecture_carry.v1"):
    descriptor = write_deterministic_npz(
        tmp_path / "carry-probe.npz", {"count": np.asarray(2, dtype=np.int32)}
    )
    return {
        "schema_version": schema,
        "state_id": "architecture-state-1",
        "pytree_descriptor_digest": digest or descriptor_digest(descriptor),
    }


def _rewrite_manifest(directory, *, update_sidecar=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    optimizer_payload = json.loads((directory / "optimizer_state.json").read_text())
    sidecar_digest = hashlib.sha256(
        (directory / "optimizer_state.npz").read_bytes()
    ).hexdigest()
    if update_sidecar:
        optimizer_payload["sidecar_digest"] = sidecar_digest
        (directory / "optimizer_state.json").write_text(
            json.dumps(optimizer_payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
    for name in manifest["files"]:
        data = (directory / name).read_bytes()
        manifest["hashes"][name] = hashlib.sha256(data).hexdigest()
        manifest["sizes"][name] = len(data)
    manifest["optimizer"]["numerical_state_sidecar_digest"] = sidecar_digest
    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    manifest["integrity"] = {
        "algorithm": "sha256",
        "manifest_digest": hashlib.sha256(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_valid_sgd_state_round_trips_and_records_both_steps(tmp_path):
    optimizer = _save(tmp_path)
    restored = load_learning_checkpoint_v3(
        tmp_path,
        optimizer=optimizer,
        parameter_layout=_layout(),
        runtime_reference="runtime-1",
    )
    assert restored.optimizer_state.envelope.step == 2
    assert int(restored.optimizer_state.arrays["step"]) == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["optimizer"]["envelope_step"] == 2
    assert manifest["optimizer"]["optimizer_numerical_state_schema_version"] == (
        "sgd_jax_state.v1"
    )


def test_strict_restore_rejects_missing_objective_identity(tmp_path):
    optimizer = _save(tmp_path)
    learning_path = tmp_path / "learning.json"
    learning = json.loads(learning_path.read_text())
    for name in (
        "objective_config",
        "resolved_objective_selection",
        "objective_descriptor",
        "objective_registry_selection",
    ):
        learning.pop(name)
    learning_path.write_text(
        json.dumps(learning, sort_keys=True, separators=(",", ":")) + "\n"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("objective")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _rewrite_manifest(tmp_path)

    with pytest.raises(
        CheckpointValidationError, match="checkpoint_objective_identity_missing"
    ):
        load_learning_checkpoint_v3(
            tmp_path,
            optimizer=optimizer,
            parameter_layout=_layout(),
        )

    historical = replace(
        _checkpoint(_state(optimizer)),
        objective_config=None,
        resolved_objective_selection=None,
        objective_descriptor=None,
        objective_registry_selection={},
    )
    migration = inspect_historical_v3_objective_alias(
        historical, source_alias="stateful_linear_mse.v1"
    )
    assert migration.status == "inspection_only_requires_recorded_migration"


def test_restore_requires_callers_expected_lifecycle_identity(tmp_path):
    optimizer = _save(tmp_path)
    checkpoint = load_learning_checkpoint_v3(
        tmp_path, optimizer=optimizer, parameter_layout=_layout()
    )
    foreign_descriptor = replace(
        checkpoint.hf_descriptor,
        tokenizer=replace(
            checkpoint.hf_descriptor.tokenizer, tokenizer_id="foreign-tokenizer"
        ),
    )
    foreign_hf = foreign_descriptor.preservation_reference()
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_hf_identity_mismatch"
    ):
        load_learning_checkpoint_v3(
            tmp_path,
            optimizer=optimizer,
            parameter_layout=_layout(),
            expected_hf_reference=foreign_hf,
            expected_hf_descriptor=foreign_descriptor,
        )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_config_identity_mismatch"
    ):
        load_learning_checkpoint_v3(
            tmp_path,
            optimizer=optimizer,
            parameter_layout=_layout(),
            expected_architecture_config_digest="foreign-config",
        )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_catalog_identity_mismatch"
    ):
        load_learning_checkpoint_v3(
            tmp_path,
            optimizer=optimizer,
            parameter_layout=_layout(),
            expected_parameter_catalog_digest="foreign-catalog",
        )

    assert (
        load_learning_checkpoint_v3(
            tmp_path,
            optimizer=optimizer,
            parameter_layout=_layout(),
            expected_hf_reference=checkpoint.hf_reference,
            expected_architecture_config_digest=checkpoint.architecture_config_digest,
            expected_parameter_catalog_digest=checkpoint.parameter_catalog_digest,
            expected_architecture_state_id="architecture-state-1",
            expected_architecture_carry_descriptor=checkpoint.architecture_carry_descriptor,
        ).architecture_state.state_id
        == "architecture-state-1"
    )


def test_custom_carry_identity_must_match_actual_descriptor_at_save(tmp_path):
    checkpoint = replace(
        _checkpoint(_state(SgdOptimizer())),
        architecture_carry_descriptor=_carry_identity(tmp_path, digest="stale"),
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_architecture_descriptor_mismatch"
    ):
        save_learning_checkpoint_v3(
            checkpoint, tmp_path / "checkpoint", optimizer=SgdOptimizer()
        )


def test_custom_carry_identity_schema_is_validated_at_save(tmp_path):
    checkpoint = replace(
        _checkpoint(_state(SgdOptimizer())),
        architecture_carry_descriptor=_carry_identity(
            tmp_path, schema="architecture_carry.legacy"
        ),
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_architecture_descriptor_mismatch"
    ):
        save_learning_checkpoint_v3(
            checkpoint, tmp_path / "checkpoint", optimizer=SgdOptimizer()
        )


def test_rehashed_carry_descriptor_cannot_escape_identity_binding(tmp_path):
    optimizer = _save(tmp_path)
    carry_descriptor = json.loads((tmp_path / "architecture_carry.json").read_text())
    carry = read_deterministic_npz(
        tmp_path / "architecture_carry.npz", carry_descriptor
    )
    carry["extra"] = np.asarray(1, dtype=np.int32)
    new_descriptor = write_deterministic_npz(tmp_path / "architecture_carry.npz", carry)
    (tmp_path / "architecture_carry.json").write_text(
        json.dumps(new_descriptor, sort_keys=True, separators=(",", ":")) + "\n"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for name in manifest["files"]:
        data = (tmp_path / name).read_bytes()
        manifest["hashes"][name] = hashlib.sha256(data).hexdigest()
        manifest["sizes"][name] = len(data)
    manifest["architecture"]["carry_descriptor_digest"] = descriptor_digest(
        new_descriptor
    )
    manifest.pop("integrity", None)
    manifest["integrity"] = {
        "algorithm": "sha256",
        "manifest_digest": hashlib.sha256(
            (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_architecture_descriptor_mismatch"
    ):
        load_learning_checkpoint_v3(
            tmp_path, optimizer=optimizer, parameter_layout=_layout()
        )


def test_envelope_step_tamper_fails_save_validation(tmp_path):
    optimizer = SgdOptimizer()
    state = _state(optimizer)
    tampered = replace(state, envelope=replace(state.envelope, step=3))
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_optimizer_step_mismatch"
    ):
        save_learning_checkpoint_v3(
            _checkpoint(tampered), tmp_path, optimizer=optimizer
        )


def test_numerical_step_tamper_fails_save_validation(tmp_path):
    optimizer = SgdOptimizer()
    state = _state(optimizer)
    tampered = replace(
        state,
        arrays={
            "step": np.asarray(3, dtype=np.int32),
            "per_parameter_steps": {"w": np.asarray(2, dtype=np.int32)},
        },
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_optimizer_step_mismatch"
    ):
        save_learning_checkpoint_v3(
            _checkpoint(tampered), tmp_path, optimizer=optimizer
        )


def test_optimizer_parameter_paths_mismatch_fails_save_validation(tmp_path):
    optimizer = SgdOptimizer()
    tampered = replace(
        _state(optimizer),
        envelope=replace(
            _state(optimizer).envelope, parameter_paths=("foreign.weight",)
        ),
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_optimizer_parameter_paths_mismatch"
    ):
        save_learning_checkpoint_v3(
            _checkpoint(tampered), tmp_path, optimizer=optimizer
        )


def test_npz_only_edit_is_rejected_by_manifest_integrity(tmp_path):
    optimizer = _save(tmp_path)
    descriptor = json.loads((tmp_path / "optimizer_state.json").read_text())[
        "numerical_state_descriptor"
    ]
    arrays = read_deterministic_npz(tmp_path / "optimizer_state.npz", descriptor)
    arrays["step"] = np.asarray(3, dtype=np.int32)
    write_deterministic_npz(tmp_path / "optimizer_state.npz", arrays)
    with pytest.raises(CheckpointValidationError, match="component hash"):
        load_learning_checkpoint_v3(
            tmp_path, optimizer=optimizer, parameter_layout=_layout()
        )


def test_rehashed_npz_edit_is_rejected_by_optimizer_consistency(tmp_path):
    optimizer = _save(tmp_path)
    descriptor = json.loads((tmp_path / "optimizer_state.json").read_text())[
        "numerical_state_descriptor"
    ]
    arrays = read_deterministic_npz(tmp_path / "optimizer_state.npz", descriptor)
    arrays["step"] = np.asarray(3, dtype=np.int32)
    write_deterministic_npz(tmp_path / "optimizer_state.npz", arrays)
    _rewrite_manifest(tmp_path, update_sidecar=True)
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_optimizer_step_mismatch"
    ):
        load_learning_checkpoint_v3(
            tmp_path, optimizer=optimizer, parameter_layout=_layout()
        )


def test_rehashed_envelope_parameter_paths_are_rejected_on_restore(tmp_path):
    optimizer = _save(tmp_path)
    optimizer_payload = json.loads((tmp_path / "optimizer_state.json").read_text())
    optimizer_payload["envelope"]["parameter_paths"] = ["foreign.weight"]
    (tmp_path / "optimizer_state.json").write_text(
        json.dumps(optimizer_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _rewrite_manifest(tmp_path)
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_optimizer_parameter_paths_mismatch"
    ):
        load_learning_checkpoint_v3(
            tmp_path, optimizer=optimizer, parameter_layout=_layout()
        )


def test_v3_write_is_atomic_and_does_not_mutate_existing_destination(tmp_path):
    optimizer = _save(tmp_path / "checkpoint")
    destination = tmp_path / "checkpoint"
    before = {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    }
    with pytest.raises(CheckpointValidationError, match="existing checkpoint"):
        save_learning_checkpoint_v3(
            _checkpoint(_state(optimizer)), destination, optimizer=optimizer
        )
    after = {
        path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()
    }
    assert before == after
    assert not tuple(tmp_path.glob(".checkpoint.tmp-*"))


def test_v3_repeated_writes_are_byte_identical(tmp_path):
    optimizer = SgdOptimizer()
    save_learning_checkpoint_v3(
        _checkpoint(_state(optimizer)), tmp_path / "one", optimizer=optimizer
    )
    save_learning_checkpoint_v3(
        _checkpoint(_state(optimizer)), tmp_path / "two", optimizer=optimizer
    )
    assert all(
        (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
        for name in (
            "parameters.npz",
            "parameters.json",
            "architecture_carry.npz",
            "architecture_carry.json",
            "optimizer_state.npz",
            "optimizer_state.json",
            "learning.json",
            "layout.json",
            "manifest.json",
        )
    )


@pytest.mark.jax
def test_restored_state_advances_both_step_identities_together(tmp_path):
    optimizer = _save(tmp_path)
    restored = load_learning_checkpoint_v3(
        tmp_path, optimizer=optimizer, parameter_layout=_layout()
    )
    _, arrays, _, _ = optimizer.apply_jax_updates(
        parameters={"w": np.asarray((1.0,), dtype=np.float32)},
        gradients={"w": np.asarray((1.0,), dtype=np.float32)},
        optimizer_array_state=restored.optimizer_state.arrays,
        update_mask={"w": True},
        config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1),
        schedule_values={},
    )
    advanced = advanced_jax_optimizer_state(restored.optimizer_state, arrays)
    validate_jax_optimizer_state(
        advanced,
        optimizer=optimizer,
        optimizer_id=optimizer.optimizer_id,
        parameter_layout=_layout(),
        descriptor=optimizer.jax_state_descriptor(_layout()),
    )
    assert advanced.envelope.step == int(advanced.arrays["step"]) == 3


@pytest.mark.jax
def test_uninterrupted_and_resumed_runs_match_both_optimizer_step_identities(tmp_path):
    optimizer = _save(tmp_path)
    resumed = load_learning_checkpoint_v3(
        tmp_path, optimizer=optimizer, parameter_layout=_layout()
    ).optimizer_state

    def advance(state):
        for _ in range(2):
            _, arrays, _, _ = optimizer.apply_jax_updates(
                parameters={"w": np.asarray((1.0,), dtype=np.float32)},
                gradients={"w": np.asarray((1.0,), dtype=np.float32)},
                optimizer_array_state=state.arrays,
                update_mask={"w": True},
                config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1),
                schedule_values={},
            )
            state = advanced_jax_optimizer_state(state, arrays)
        return state

    uninterrupted = advance(_state(optimizer))
    resumed = advance(resumed)
    assert uninterrupted.envelope.step == resumed.envelope.step == 4
    assert int(uninterrupted.arrays["step"]) == int(resumed.arrays["step"]) == 4


def test_generic_v3_code_has_no_sgd_step_path_assumption():
    source = (
        Path(__file__).parents[1] / "src/radjax_student/checkpoints/v3.py"
    ).read_text()
    assert 'arrays["step"]' not in source


def test_non_sgd_optimizer_owns_a_different_step_representation(tmp_path):
    class MomentumOptimizer(SgdOptimizer):
        optimizer_id = "momentum.v1"

        def capability_profile(self):
            return OptimizerCapabilityProfile(
                self.optimizer_id, 1, super().capability_profile().capabilities
            )

        def jax_state_descriptor(self, parameter_layout):
            return JaxOptimizerStateDescriptor(
                self.optimizer_id,
                "optimizer.jax_execution_v1",
                "momentum_jax_state.v1",
                parameter_layout.digest(),
                (("clock",), ("momentum",)),
            )

        def validate_jax_state(self, *, arrays, descriptor, envelope):
            assert descriptor.optimizer_id == self.optimizer_id
            assert int(arrays["clock"][0]) == envelope.step
            assert tuple(arrays["momentum"].shape) == (1,)

    optimizer = MomentumOptimizer(optimizer_id="momentum.v1")
    envelope = OptimizerState(optimizer.optimizer_id, _layout().logical_paths, step=2)
    descriptor = optimizer.jax_state_descriptor(_layout())
    state = JaxOptimizerState(
        envelope,
        descriptor,
        {
            "clock": np.asarray((2,), dtype=np.int32),
            "momentum": np.asarray((0.5,), dtype=np.float32),
        },
    )
    save_learning_checkpoint_v3(_checkpoint(state), tmp_path, optimizer=optimizer)
    restored = load_learning_checkpoint_v3(
        tmp_path, optimizer=optimizer, parameter_layout=_layout()
    )
    assert (
        int(restored.optimizer_state.arrays["clock"][0])
        == restored.optimizer_state.envelope.step
    )
