"""P5.6 corridor-only deterministic checkpoint evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.behavior import (  # noqa: E402
    CorridorPassError,
    CorridorRunBindingV1,
    materialize_behavioral_batches_v1,
    replay_corridor_pass_v1,
    run_corridor_pass_v1,
)
from radjax_student.contracts import (  # noqa: E402
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.optimizers import (  # noqa: E402
    OptimizerConfig,
    OptimizerState,
    SgdOptimizer,
)
from tests.test_p5_4_behavior_materialization import _projection  # noqa: E402

pytestmark = pytest.mark.jax


def _binding() -> CorridorRunBindingV1:
    return CorridorRunBindingV1(
        contract_commit="a33a0a0d90a57e5cd67155c01f054d3e7a04dc0d",
        tome_commit="8508b1351d0ed8d6a3a14049e4d6f8a849c33cf1",
        accepted_receipt_identity="sha256:" + "1" * 64,
        language_binding_digest="sha256:" + "2" * 64,
        hf_projection_identity="sha256:" + "3" * 64,
        behavioral_source_identity="sha256:" + "4" * 64,
        behavioral_authority_digest="sha256:" + "5" * 64,
        architecture_config_identity="sha256:" + "6" * 64,
        split_identity="sha256:" + "7" * 64,
        corridor_objective_policy_id="radjax.student.behavior_objective.v1",
        reduction_id="assignment_attention_weighted_mean.v1",
    )


def _setup():
    projection = _admitted_projection()
    materialization = materialize_behavioral_batches_v1(projection)
    batch = materialization.training_corridor
    layout = ParameterTreeLayout(
        "neutral.proof",
        (
            ParameterTreeLayoutEntry(
                "head.bias", ("head", "bias"), (4,), "float32", "output_head"
            ),
        ),
    )
    optimizer = SgdOptimizer()
    config = OptimizerConfig(optimizer.optimizer_id, learning_rate=0.05)
    state = optimizer.initialize_jax_state(
        config=config,
        parameter_layout=layout,
        optimizer_state=OptimizerState(optimizer.optimizer_id, layout.logical_paths),
    )
    parameters = {
        "head": {"bias": jnp.asarray([0.0, 0.1, 0.2, 0.4], dtype=jnp.float32)}
    }

    def forward(values, input_ids, attention_mask):
        del attention_mask
        return jnp.broadcast_to(values["head"]["bias"], (*input_ids.shape, 4))

    return dict(
        batch=batch,
        binding=_binding().__class__(
            **{
                **_binding().to_dict(),
                "behavioral_source_identity": (
                    materialization.split.behavioral_source_identity
                ),
                "split_identity": materialization.split.split_identity,
            }
        ),
        materialization=materialization,
        projection=projection,
        parameters=parameters,
        parameter_layout=layout,
        optimizer=optimizer,
        optimizer_config=config,
        optimizer_state=state,
        forward=forward,
    )


def _admitted_projection():
    """Mint the compact test projection through the verified-admission factory."""

    raw = _projection()
    from radjax_student.artifacts import native_v3_v6

    descriptor = SimpleNamespace(
        language_binding_digest="sha256:" + "b" * 64,
        behavioral_source_identity=raw.behavioral_source_identity,
        behavioral_authority_digest=raw.behavioral_authority_digest,
        package_semantic_identity=raw.package_semantic_identity,
        composition_digest=raw.composition_digest,
        authority_resources=tuple(
            SimpleNamespace(resource_id=resource_id, role=role)
            for role, resource_id in native_v3_v6._AUTHORITY_RESOURCE_IDS.items()
        ),
    )
    with patch.multiple(
        native_v3_v6,
        validate_and_resolve_student_consumption=lambda *_args, **_kwargs: (
            SimpleNamespace(ok=True, descriptor=descriptor)
        ),
        resolve_student_language_binding=lambda *_args, **_kwargs: SimpleNamespace(
            canonical_binding_digest=descriptor.language_binding_digest
        ),
        _language_projection=lambda _descriptor: raw.language,
        _json_resource_projection=lambda _artifact, _descriptor, resource_id: (
            raw.authority_reference
            if resource_id == "authority_reference/default"
            else raw.corridor_mode_table
        ),
        _multipart_projection=lambda _artifact, resource_id: (
            raw.target_shard
            if resource_id == "target_shard/default"
            else raw.corridor_assignment
        ),
        _jsonl_records=lambda _artifact, resource_id: (
            raw.example_registry
            if resource_id == "example_registry/default"
            else raw.selected_passports
        ),
        _m7_records=lambda *_args, **_kwargs: raw.selected_exemplar_payloads,
    ):
        return native_v3_v6.open_native_v3_v6_behavioral_projection("test-artifact")


def test_p5_6_corridor_pass_is_training_only_finite_and_changes_parameters():
    values = _setup()
    result = run_corridor_pass_v1(**values)
    assert result.loss > 0 and result.gradient_norm > 0
    assert result.changed_parameter_paths == ("head.bias",)
    assert result.checkpoint.pass_id == "corridor_v1"
    assert result.checkpoint.cursor == len(result.ordered_coordinates)
    assert {item[0] for item in result.ordered_coordinates} <= set(
        values["batch"].example_ids
    )
    assert result.checkpoint.binding.identity


def test_p5_6_corridor_order_and_checkpoint_replay_are_exact():
    values = _setup()
    result = run_corridor_pass_v1(**values)
    replay = replay_corridor_pass_v1(expected=result.checkpoint, **values)
    assert replay.ordered_coordinates == result.ordered_coordinates
    assert replay.checkpoint.identity == result.checkpoint.identity


def test_p5_6_held_out_or_changed_binding_fails_closed():
    values = _setup()
    held_out = materialize_behavioral_batches_v1(_projection()).held_out_corridor
    with pytest.raises(CorridorPassError, match="held-out"):
        run_corridor_pass_v1(**{**values, "batch": held_out})
    result = run_corridor_pass_v1(**values)
    with pytest.raises(CorridorPassError, match="materialization continuity mismatch"):
        replay_corridor_pass_v1(
            expected=result.checkpoint,
            **{
                **values,
                "binding": _binding().__class__(
                    **{**_binding().to_dict(), "split_identity": "sha256:" + "8" * 64}
                ),
            },
        )


def test_p5_6_optimizer_state_from_another_backend_identity_fails_closed():
    values = _setup()
    other = SgdOptimizer(optimizer_id="other.sgd")
    layout = values["parameter_layout"]
    other_state = other.initialize_jax_state(
        config=OptimizerConfig(other.optimizer_id, learning_rate=0.05),
        parameter_layout=layout,
        optimizer_state=OptimizerState(other.optimizer_id, layout.logical_paths),
    )
    with pytest.raises(CorridorPassError, match="supplied optimizer"):
        run_corridor_pass_v1(**{**values, "optimizer_state": other_state})


def test_p5_6_rejects_unsealed_or_forged_materialization_identity():
    values = _setup()
    with pytest.raises(CorridorPassError, match="sealed P5.4 materialization"):
        run_corridor_pass_v1(**{**values, "materialization": object()})
    forged = _binding().__class__(
        **{
            **values["binding"].to_dict(),
            "split_identity": "sha256:" + "f" * 64,
        }
    )
    with pytest.raises(CorridorPassError, match="materialization continuity mismatch"):
        run_corridor_pass_v1(**{**values, "binding": forged})


def test_p5_6_rejects_descriptor_forged_to_subset_passport_authority_before_jax():
    values = _setup()
    materialization = values["materialization"]
    held_out = replace(
        materialization.held_out_exemplars,
        input_ids=materialization.held_out_exemplars.input_ids[:1],
        attention_mask=materialization.held_out_exemplars.attention_mask[:1],
        example_ids=materialization.held_out_exemplars.example_ids[:1],
        selected_example_indices=(
            materialization.held_out_exemplars.selected_example_indices[:1]
        ),
        selected_positions=materialization.held_out_exemplars.selected_positions[:1],
        sparse_targets=materialization.held_out_exemplars.sparse_targets[:1],
        passports=materialization.held_out_exemplars.passports[:1],
    )
    held_out_keys = tuple(
        (
            str(passport["selected_example_id"]),
            passport["selected_position"],
            str(passport["corridor_fingerprint_id"]),
        )
        for passport in held_out.passports
    )
    omitted = (
        str(materialization.held_out_exemplars.passports[-1]["selected_example_id"]),
        materialization.held_out_exemplars.passports[-1]["selected_position"],
        str(
            materialization.held_out_exemplars.passports[-1]["corridor_fingerprint_id"]
        ),
    )
    subset_keys = tuple(
        key
        for key in materialization.descriptor.authoritative_exemplar_passport_keys
        if key != omitted
    )
    held_out_identity = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {"passports": held_out_keys}, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    forged = replace(
        materialization,
        held_out_exemplars=held_out,
        descriptor=replace(
            materialization.descriptor,
            held_out_exemplar_identity=held_out_identity,
            authoritative_exemplar_passport_keys=subset_keys,
        ),
    )
    with pytest.raises(CorridorPassError, match="passport authority mismatch"):
        run_corridor_pass_v1(
            **{
                **values,
                "materialization": forged,
                "batch": forged.training_corridor,
            }
        )
    reconstructed = replace(values["projection"])
    with pytest.raises(CorridorPassError, match="requires admitted P5.3 projection"):
        run_corridor_pass_v1(
            **{
                **values,
                "materialization": forged,
                "batch": forged.training_corridor,
                "projection": reconstructed,
            }
        )
