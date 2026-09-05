"""M2.5 generic Student lifecycle proof for configurable Mamba-2."""

from __future__ import annotations

# JAX availability is checked before importing JAX-bearing production modules.
# ruff: noqa: E402, I001
import pytest

jax = pytest.importorskip("jax", reason="Mamba-2 lifecycle tests require JAX")
jnp = pytest.importorskip("jax.numpy", reason="Mamba-2 lifecycle tests require JAX")

from radjax_student.learning.jax_batch import FiniteJsonJaxBatchMaterializer
from radjax_student.learning.jax_core import build_registered_jax_loss_fn
from tests.support.mamba2_learning import (
    all_finite,
    assembled,
    batch,
    execute,
    tree_allclose,
    tree_changed,
)

pytestmark = pytest.mark.jax


def test_v512_t8_configuration_executes_through_eager_and_jit_lifecycles():
    eager = assembled("eager")
    compiled = assembled("jit")
    eager_before = eager.loop_executor.lifecycle
    compiled_before = compiled.loop_executor.lifecycle
    eager_execution = execute(eager, batch())
    compiled_execution = execute(compiled, batch())

    for execution, before, after, is_compiled in (
        (
            eager_execution,
            eager_before,
            eager.loop_executor.lifecycle,
            False,
        ),
        (
            compiled_execution,
            compiled_before,
            compiled.loop_executor.lifecycle,
            True,
        ),
    ):
        assert execution.result.status == "pass"
        assert execution.result.loss is not None
        assert execution.result.loss.loss > 0.0
        assert all_finite(execution.gradients)
        assert tree_changed(before.parameters, after.parameters)
        assert tree_changed(before.architecture_carry, after.architecture_carry)
        assert all_finite(after.architecture_carry)
        assert after.learning_state.global_step == 1
        assert after.optimizer_state.envelope.step == 1
        assert execution.runtime_result.compiled is is_compiled
        assert execution.runtime_result.callable_reference is not None
        assert (
            execution.runtime_result.callable_reference.callable_id
            == "radjax.learning.generic_jax_step"
        )

    assert tree_allclose(
        eager.loop_executor.lifecycle.parameters,
        compiled.loop_executor.lifecycle.parameters,
    )
    assert tree_allclose(
        eager.loop_executor.lifecycle.architecture_carry,
        compiled.loop_executor.lifecycle.architecture_carry,
    )
    second = execute(compiled, batch((5, 3, 7, 1)))
    assert second.result.status == "pass"
    assert second.runtime_result.compiled is True
    assert second.runtime_result.callable_reference == (
        compiled_execution.runtime_result.callable_reference
    )
    assert second.runtime_result.output_metadata["rng_bridge"]["global_step"] == 1


def test_two_optimizer_steps_match_independently_built_reference_schedule():
    actual = assembled("eager")
    reference = assembled("eager")
    first_batch = batch()
    second_batch = batch((5, 3, 7, 1))
    actual_first = execute(actual, first_batch)
    actual_second = execute(actual, second_batch)
    reference_first = execute(reference, first_batch)
    reference_second = execute(reference, second_batch)

    assert actual_first.result.status == actual_second.result.status == "pass"
    assert reference_first.result.status == reference_second.result.status == "pass"
    assert tree_allclose(
        actual.loop_executor.lifecycle.parameters,
        reference.loop_executor.lifecycle.parameters,
    )
    assert tree_allclose(
        actual.loop_executor.lifecycle.architecture_carry,
        reference.loop_executor.lifecycle.architecture_carry,
    )
    assert actual.loop_executor.lifecycle.learning_state.global_step == 2
    assert actual.loop_executor.lifecycle.optimizer_state.envelope.step == 2


def test_learning_step_stop_gradient_covers_both_persistent_state_families():
    assembled_lifecycle = assembled("eager")
    lifecycle = assembled_lifecycle.lifecycle
    loss_fn = build_registered_jax_loss_fn(
        architecture=lifecycle.architecture,
        objective_selection=lifecycle.objective_selection,
        objective_config=lifecycle.objective_config,
        objective_descriptor=lifecycle.objective_descriptor,
        resolved_selection=lifecycle.resolved_objective_selection,
        architecture_config=lifecycle.architecture_config,
    )
    materialized = FiniteJsonJaxBatchMaterializer().materialize(batch())

    def loss_from_carry(carry):
        return loss_fn(lifecycle.parameters, carry, materialized, None)[0]

    gradients = jax.grad(loss_from_carry)(lifecycle.architecture_carry)
    assert all(
        bool(jnp.array_equal(leaf, jnp.zeros_like(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradients)
    )
