"""P5.6 corridor-only deterministic checkpoint evidence."""

from __future__ import annotations

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
    batch = materialize_behavioral_batches_v1(_projection()).training_corridor
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
        binding=_binding(),
        parameters=parameters,
        parameter_layout=layout,
        optimizer=optimizer,
        optimizer_config=config,
        optimizer_state=state,
        forward=forward,
    )


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
    with pytest.raises(CorridorPassError, match="replay mismatch"):
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
