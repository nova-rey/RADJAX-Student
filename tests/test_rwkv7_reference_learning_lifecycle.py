"""Focused P4.5 proof for RWKV-7 through the generic JAX learning lifecycle."""

from __future__ import annotations

# JAX availability is checked before importing JAX-bearing production modules.
# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax", reason="RWKV-7 lifecycle tests require JAX")
jnp = pytest.importorskip("jax.numpy", reason="RWKV-7 lifecycle tests require JAX")

from radjax_student.architecture.rwkv7_reference import (
    RWKV7ReferencePlugin,
    reference_architecture_config,
)
from radjax_student.learning import LearningBatch
from radjax_student.learning.jax_batch import FiniteJsonJaxBatchMaterializer
from radjax_student.learning.jax_core import (
    build_registered_jax_loss_fn,
    build_value_and_grad_fn,
)
from tests.support.rwkv7_learning import (
    TARGETS,
    TOKENS,
    all_finite,
    assembled,
    batch,
    execute,
    tree_allclose,
    tree_changed,
)

pytestmark = pytest.mark.jax

GRADIENT_BOUNDARY = (
    "Gradients differentiate through multiple token positions within one fixture "
    "sequence.",
    "Carry returned from one learning step can seed a later learning step.",
    "Carry crossing separate learning-step boundaries is stop-gradient state.",
    "Full cross-step BPTT, truncated-BPTT scheduling, and long-context recurrent "
    "training are not proven.",
)


def test_rwkv_batch_validation_accepts_only_the_finite_json_tiny_domain() -> None:
    plugin = RWKV7ReferencePlugin()
    config = reference_architecture_config()

    assert plugin.validate_batch(batch(), config).status == "pass"
    for malformed in (
        ((1, 7, 3),),
        ((1, 7, 3, 16),),
        ((1, 7, 3, True),),
    ):
        malformed_batch = LearningBatch(
            "p45-rwkv7",
            inputs={"token_ids": [list(malformed[0])]},
            targets={"token_ids": [list(TARGETS)]},
        )
        validation = plugin.validate_batch(malformed_batch, config)
        assert validation.status == "fail"
        assert validation.blockers[0].code == "architecture_batch_incompatible"


def test_eager_and_jit_registered_lifecycles_agree_and_advance_state() -> None:
    eager = assembled("eager")
    compiled = assembled("jit")
    eager_before = eager.lifecycle
    compiled_before = compiled.lifecycle

    eager_execution = execute(eager, batch())
    compiled_execution = execute(compiled, batch())
    eager_after = eager.loop_executor.lifecycle
    compiled_after = compiled.loop_executor.lifecycle

    for execution, before, after, is_compiled in (
        (eager_execution, eager_before, eager_after, False),
        (compiled_execution, compiled_before, compiled_after, True),
    ):
        assert execution.result.status == "pass"
        assert execution.result.loss is not None
        assert execution.result.loss.loss > 0.0
        assert set(execution.objective_metrics) == {
            "objective.sparse_cross_entropy",
            "objective.token_accuracy",
        }
        assert all_finite(execution.gradients)
        assert execution.result.changed_parameter_paths
        assert tree_changed(before.parameters, after.parameters)
        assert tree_changed(before.architecture_carry, after.architecture_carry)
        assert all_finite(after.architecture_carry)
        assert after.learning_state.global_step == before.learning_state.global_step + 1
        assert (
            after.learning_state.optimizer_step
            == before.learning_state.optimizer_step + 1
        )
        assert (
            after.optimizer_state.envelope.step
            == before.optimizer_state.envelope.step + 1
        )
        assert execution.runtime_result.compiled is is_compiled
        assert execution.runtime_result.callable_reference is not None
        assert (
            execution.runtime_result.callable_reference.callable_id
            == "radjax.learning.generic_jax_step"
        )
        assert execution.runtime_result.output_metadata["rng_bridge"] == {
            "schema_version": "runtime_jax_key_bridge.v1",
            "prng_implementation": "threefry2x32",
            "stream": "dropout",
            "slot": "dropout",
            "global_step": 0,
            "micro_step": 0,
            "invocation_index": 0,
        }

    assert eager_execution.runtime_result.callable_reference == (
        compiled_execution.runtime_result.callable_reference
    )
    assert (
        eager_execution.runtime_result.prepared_execution_digest
        != compiled_execution.runtime_result.prepared_execution_digest
    )
    assert tree_allclose(eager_after.parameters, compiled_after.parameters)
    assert tree_allclose(
        eager_after.architecture_carry, compiled_after.architecture_carry
    )
    assert eager_execution.result.loss.loss == pytest.approx(
        compiled_execution.result.loss.loss, rel=1e-5, abs=2e-5
    )

    later_before = eager.loop_executor.lifecycle
    assert tree_allclose(
        later_before.architecture_carry, eager_after.architecture_carry
    )
    later = execute(eager, batch((5, 3, 7, 1)))
    later_after = eager.loop_executor.lifecycle
    assert later.result.status == "pass"
    assert later_after.learning_state.global_step == 2
    assert later_after.optimizer_state.envelope.step == 2
    assert later.runtime_result.output_metadata["rng_bridge"]["global_step"] == 1
    assert later.runtime_result.output_metadata["rng_bridge"]["invocation_index"] == 1


def test_within_sequence_gradients_and_cross_step_carry_boundary() -> None:
    lifecycle_result = assembled("eager")
    lifecycle = lifecycle_result.lifecycle
    loss_fn = build_registered_jax_loss_fn(
        architecture=lifecycle.architecture,
        objective_selection=lifecycle.objective_selection,
        objective_config=lifecycle.objective_config,
        objective_descriptor=lifecycle.objective_descriptor,
        resolved_selection=lifecycle.resolved_objective_selection,
    )
    materializer = FiniteJsonJaxBatchMaterializer()
    base_batch = materializer.materialize(batch())
    value_and_grad = build_value_and_grad_fn(loss_fn)

    def gradient_for(tokens: tuple[int, ...]):
        materialized_batch = materializer.materialize(batch(tokens))
        (_, _), gradients = value_and_grad(
            lifecycle.parameters,
            lifecycle.architecture_carry,
            materialized_batch,
            None,
        )
        return gradients

    base = gradient_for(TOKENS)
    changed_first = gradient_for((2, 7, 3, 5))
    changed_third = gradient_for((1, 7, 4, 5))
    assert all_finite(base)
    assert tree_changed(base, changed_first)
    assert tree_changed(base, changed_third)

    def loss_from_input_carry(carry):
        return loss_fn(lifecycle.parameters, carry, base_batch, None)[0]

    def sum_of_returned_carry(carry):
        _, auxiliary = loss_fn(lifecycle.parameters, carry, base_batch, None)
        return sum(
            jnp.sum(leaf)
            for leaf in jax.tree_util.tree_leaves(auxiliary.updated_architecture_carry)
        )

    for carry_gradient in (
        jax.grad(loss_from_input_carry)(lifecycle.architecture_carry),
        jax.grad(sum_of_returned_carry)(lifecycle.architecture_carry),
    ):
        assert all(
            bool(jnp.array_equal(leaf, jnp.zeros_like(leaf)))
            for leaf in jax.tree_util.tree_leaves(carry_gradient)
        )
