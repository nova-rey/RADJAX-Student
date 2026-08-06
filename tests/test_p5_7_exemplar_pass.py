"""P5.7 sequential exemplar continuation evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.behavior import (  # noqa: E402
    ExemplarPassError,
    ExemplarRunBindingV1,
    materialize_behavioral_batches_v1,
    replay_exemplar_pass_v1,
    run_corridor_pass_v1,
    run_exemplar_pass_v1,
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
from tests.test_p5_6_corridor_pass import _binding  # noqa: E402

pytestmark = pytest.mark.jax


def _setup():
    batches = materialize_behavioral_batches_v1(_projection())
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

    corridor = run_corridor_pass_v1(
        batch=batches.training_corridor,
        binding=_binding(),
        parameters=parameters,
        parameter_layout=layout,
        optimizer=optimizer,
        optimizer_config=config,
        optimizer_state=state,
        forward=forward,
    ).checkpoint
    return dict(
        predecessor=corridor,
        batch=batches.training_exemplars,
        binding=ExemplarRunBindingV1.from_corridor_checkpoint(corridor),
        parameter_layout=layout,
        optimizer=optimizer,
        optimizer_config=config,
        forward=forward,
    )


def test_p5_7_resumes_corridor_then_updates_exemplars_to_final_checkpoint():
    values = _setup()
    result = run_exemplar_pass_v1(**values)
    assert result.loss > 0 and result.gradient_norm > 0
    assert result.changed_parameter_paths == ("head.bias",)
    assert result.checkpoint.pass_id == "exemplar_v1"
    assert result.checkpoint.cursor == len(result.ordered_passport_keys)
    assert (
        result.checkpoint.binding.predecessor_checkpoint_identity
        == values["predecessor"].identity
    )


def test_p5_7_rejects_non_exemplar_batch_before_accessing_batch_fields():
    values = _setup()
    corridor = materialize_behavioral_batches_v1(_projection()).training_corridor
    with pytest.raises(ExemplarPassError, match="requires a neutral exemplar batch"):
        run_exemplar_pass_v1(**{**values, "batch": corridor})


def test_p5_7_exact_replay_preserves_model_optimizer_and_checkpoint_identity():
    values = _setup()
    result = run_exemplar_pass_v1(**values)
    replay = replay_exemplar_pass_v1(expected=result.checkpoint, **values)
    assert replay.checkpoint.identity == result.checkpoint.identity
    assert replay.checkpoint.optimizer_state == result.checkpoint.optimizer_state
    for observed, expected in zip(
        jax.tree_util.tree_leaves(replay.checkpoint.parameters),
        jax.tree_util.tree_leaves(result.checkpoint.parameters),
        strict=True,
    ):
        assert bool(jnp.array_equal(observed, expected))


@pytest.mark.parametrize(
    "alteration, field, message",
    [
        ("predecessor", None, "predecessor checkpoint identity mismatch"),
        ("continuity", "behavioral_source_identity", "continuation authority mismatch"),
        ("continuity", "split_identity", "continuation authority mismatch"),
        (
            "continuity",
            "architecture_config_identity",
            "continuation authority mismatch",
        ),
        (
            "continuity",
            "corridor_objective_policy_id",
            "continuation authority mismatch",
        ),
    ],
)
def test_p5_7_wrong_predecessor_or_changed_continuity_fails_closed(
    alteration, field, message
):
    values = _setup()
    if alteration == "predecessor":
        values["binding"] = replace(
            values["binding"], predecessor_checkpoint_identity="sha256:" + "0" * 64
        )
    else:
        assert field is not None
        values["binding"] = replace(values["binding"], **{field: "changed.v1"})
    with pytest.raises(ExemplarPassError, match=message):
        run_exemplar_pass_v1(**values)


def test_p5_7_held_out_reordered_passports_and_mixed_objective_fail_closed():
    values = _setup()
    held_out = materialize_behavioral_batches_v1(_projection()).held_out_exemplars
    with pytest.raises(ExemplarPassError, match="held-out"):
        run_exemplar_pass_v1(**{**values, "batch": held_out})
    batch = values["batch"]
    later_passport = {
        **batch.passports[0],
        "selected_position": int(batch.passports[0]["selected_position"]) + 1,
        "corridor_fingerprint_id": "later-fingerprint",
    }
    reordered = replace(
        batch,
        passports=(later_passport, batch.passports[0]),
        sparse_targets=(batch.sparse_targets[0], batch.sparse_targets[0]),
        selected_positions=jnp.asarray(
            [batch.selected_positions[0] + 1, batch.selected_positions[0]]
        ),
        selected_example_indices=jnp.asarray(
            [batch.selected_example_indices[0], batch.selected_example_indices[0]]
        ),
        input_ids=jnp.asarray([batch.input_ids[0], batch.input_ids[0]]),
        attention_mask=jnp.asarray([batch.attention_mask[0], batch.attention_mask[0]]),
    )
    with pytest.raises(ExemplarPassError, match="reordered"):
        run_exemplar_pass_v1(**{**values, "batch": reordered})
    with pytest.raises(
        ExemplarPassError, match="unsupported exemplar objective policy"
    ):
        ExemplarRunBindingV1.from_corridor_checkpoint(values["predecessor"]).__class__(
            **{
                **values["binding"].to_dict(),
                "exemplar_objective_policy_id": "corridor_objective_v1",
            }
        )
