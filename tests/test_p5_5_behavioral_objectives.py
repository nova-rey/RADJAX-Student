"""Independent P5.5 objective and scoped-SGD qualification evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.behavior import (  # noqa: E402
    BehavioralObjectiveError,
    corridor_objective_v1,
    exemplar_coarse_cross_entropy_v1,
    materialize_behavioral_batches_v1,
)
from radjax_student.checkpoints import (  # noqa: E402
    LearningCheckpoint,
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from radjax_student.contracts import (  # noqa: E402
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.learning import LearningState  # noqa: E402
from radjax_student.optimizers import (  # noqa: E402
    OptimizerConfig,
    OptimizerContractError,
    OptimizerState,
    SgdOptimizer,
    advanced_jax_optimizer_state,
    apply_verified_jax_updates,
    require_finite_jax_gradients,
)
from tests.test_p5_4_behavior_materialization import _projection  # noqa: E402

pytestmark = pytest.mark.jax


def _batches():
    return materialize_behavioral_batches_v1(_projection())


def _logits(batch):
    return (
        jnp.arange(np.prod((*batch.input_ids.shape, 4)), dtype=jnp.float32).reshape(
            (*batch.input_ids.shape, 4)
        )
        / 10.0
    )


def test_p5_5_corridor_statistics_are_exact_inclusive_and_jax_differentiable():
    batch = _batches().training_corridor
    logits = _logits(batch)
    eager_loss, eager_metrics = corridor_objective_v1(logits, batch)
    compiled = jax.jit(lambda values: corridor_objective_v1(values, batch))
    compiled_loss, compiled_metrics = compiled(logits)
    gradient = jax.grad(lambda values: corridor_objective_v1(values, batch)[0])(logits)

    assert float(eager_loss) == pytest.approx(float(compiled_loss))
    assert np.allclose(np.asarray(gradient), np.asarray(gradient))
    assert float(eager_metrics["corridor.all_statistics.inside_rate"]) == 0.0
    assert set(compiled_metrics) == {
        "corridor.loss",
        "corridor.all_statistics.inside_rate",
        *(
            value
            for name in (
                "entropy",
                "top1_margin",
                "top8_mass",
                "top32_mass",
                "tail_mass",
            )
            for value in (
                f"corridor.{name}.inside_rate",
                f"corridor.{name}.mean_squared_distance",
            )
        ),
    }

    bounds = batch.mode_bounds.statistics_by_mode[0]
    assert bounds["entropy"].minimum == bounds["entropy"].maximum == 0.0
    assert float(eager_metrics["corridor.entropy.mean_squared_distance"]) > 0.0


def test_p5_5_exemplar_coarse_tail_is_logsumexp_not_bucket_distribution():
    batch = _batches().training_exemplars
    logits = _logits(batch)
    loss, metrics = exemplar_coarse_cross_entropy_v1(logits, batch)
    compiled_loss, _ = jax.jit(
        lambda values: exemplar_coarse_cross_entropy_v1(values, batch)
    )(logits)
    gradient = jax.grad(
        lambda values: exemplar_coarse_cross_entropy_v1(values, batch)[0]
    )(logits)

    row = np.asarray(logits)[0, int(batch.selected_positions[0])]
    log_probs = row - np.logaddexp.reduce(row)
    target = batch.sparse_targets[0]
    inactive = np.ones(len(row), dtype=bool)
    inactive[target.token_ids] = False
    expected = -(
        np.dot(target.probabilities, log_probs[target.token_ids])
        + target.aggregate_tail_mass * np.logaddexp.reduce(log_probs[inactive])
    )
    assert float(loss) == pytest.approx(float(expected))
    assert float(loss) == pytest.approx(float(compiled_loss))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(metrics["exemplar.coarse_cross_entropy"]) == pytest.approx(float(loss))

    broken = type(target)(target.token_ids, target.probabilities, 0.2)
    invalid_batch = type(batch)(
        batch.partition,
        batch.example_ids,
        batch.input_ids,
        batch.attention_mask,
        batch.selected_example_indices,
        batch.selected_positions,
        (broken,),
        batch.passports,
    )
    with pytest.raises(BehavioralObjectiveError, match="sum to one"):
        exemplar_coarse_cross_entropy_v1(logits, invalid_batch)


def _layout() -> ParameterTreeLayout:
    return ParameterTreeLayout(
        "p5.5.objectives",
        (
            ParameterTreeLayoutEntry(
                "head.weight", ("head", "weight"), (1,), "float32", "output_head"
            ),
            ParameterTreeLayoutEntry(
                "trunk.weight", ("trunk", "weight"), (1,), "float32", "recurrent_block"
            ),
        ),
    )


def _qualify_sgd_for_behavioral_objective(*, batch, objective, checkpoint_root):
    """Exercise one behavioral objective through every production SGD promise."""

    logits = _logits(batch)
    loss_fn = lambda values: objective(values, batch)[0]  # noqa: E731
    # JIT execution and repeat autodiff are each performed for this one
    # objective; no mixed synthetic surrogate stands in for either family.
    jitted_loss = jax.jit(loss_fn)
    assert float(jitted_loss(logits)) == pytest.approx(float(loss_fn(logits)))
    first = jax.grad(loss_fn)(logits)
    second = jax.grad(loss_fn)(logits)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert np.all(np.isfinite(np.asarray(first)))
    gradient = float(
        first[
            0,
            int(batch.selected_positions[0])
            if hasattr(batch, "selected_positions")
            else int(batch.positions[0]),
            0,
        ]
    )
    optimizer, layout = SgdOptimizer(), _layout()
    envelope = OptimizerState(optimizer.optimizer_id, layout.logical_paths)
    state = optimizer.initialize_jax_state(
        config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.01),
        parameter_layout=layout,
        optimizer_state=envelope,
    )
    parameters = {
        "head": {"weight": jnp.asarray([2.0])},
        "trunk": {"weight": jnp.asarray([3.0])},
    }
    gradients = {
        "head": {"weight": jnp.asarray([gradient])},
        "trunk": {"weight": jnp.asarray([gradient])},
    }
    updated, arrays, changed, metrics = apply_verified_jax_updates(
        optimizer=optimizer,
        parameters=parameters,
        gradients=gradients,
        optimizer_array_state=state.arrays,
        update_mask=layout.update_mask(parameters, ("head.weight",)),
        config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.01),
        schedule_values={"learning_rate": 0.25},
    )
    require_finite_jax_gradients(metrics)
    assert float(updated["head"]["weight"][0]) == pytest.approx(2.0 - 0.25 * gradient)
    assert float(updated["trunk"]["weight"][0]) == 3.0
    assert bool(changed["head"]["weight"]) and not bool(changed["trunk"]["weight"])
    advanced = advanced_jax_optimizer_state(state, arrays)
    assert advanced.envelope.step == 1 and int(advanced.arrays["step"]) == 1
    with pytest.raises(OptimizerContractError, match="finite"):
        apply_verified_jax_updates(
            optimizer=optimizer,
            parameters=parameters,
            gradients={
                "head": {"weight": jnp.asarray([jnp.nan])},
                "trunk": {"weight": jnp.asarray([jnp.inf])},
            },
            optimizer_array_state=state.arrays,
            update_mask=layout.update_mask(parameters, ("head.weight",)),
            config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.01),
            schedule_values={"learning_rate": 0.25},
        )
    # The rejecting wrapper returns no candidate parameter/state tree; the
    # original values remain the only caller-visible state after NaN/Inf.
    assert float(parameters["head"]["weight"][0]) == 2.0
    assert int(state.arrays["step"]) == 0

    # The stable envelope is JSON serializable and replays through the public
    # continuation checkpoint format without serializing JAX implementation leaves.
    assert json.loads(json.dumps(advanced.envelope.to_dict()))["step"] == 1
    checkpoint = LearningCheckpoint(
        "p5.5.sgd",
        LearningState("p5.5", optimizer_step=1),
        None,
        advanced.envelope,
        {"head.weight": 1.0, "trunk.weight": 2.0},
        {"cursor": 1},
        {},
        {},
    )
    saved = save_learning_checkpoint(checkpoint, checkpoint_root / "checkpoint")
    replayed = load_learning_checkpoint(
        checkpoint_root / "checkpoint", runtime_reference="p5.5.sgd"
    )
    assert replayed.optimizer_state == saved.optimizer_state
    assert replayed.learning_state.optimizer_step == replayed.optimizer_state.step == 1


def test_p5_5_sgd_qualifies_corridor_objective_independently(tmp_path):
    _qualify_sgd_for_behavioral_objective(
        batch=_batches().training_corridor,
        objective=corridor_objective_v1,
        checkpoint_root=tmp_path,
    )


def test_p5_5_sgd_qualifies_exemplar_objective_independently(tmp_path):
    _qualify_sgd_for_behavioral_objective(
        batch=_batches().training_exemplars,
        objective=exemplar_coarse_cross_entropy_v1,
        checkpoint_root=tmp_path,
    )
