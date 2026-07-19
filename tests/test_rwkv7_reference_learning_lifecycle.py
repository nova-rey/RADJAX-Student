"""Focused P4.5 proof for RWKV-7 through the generic JAX learning lifecycle."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.architecture.rwkv7_reference import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    RWKV7ReferencePlugin,
    reference_architecture_config,
    register_rwkv7_reference,
)
from radjax_student.contracts import ObjectiveConfig, ObjectiveScope, UpdateScope
from radjax_student.learning import (
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningBatch,
    LearningState,
    assemble_jax_learning_lifecycle,
)
from radjax_student.learning.jax_batch import FiniteJsonJaxBatchMaterializer
from radjax_student.learning.jax_core import (
    build_registered_jax_loss_fn,
    build_value_and_grad_fn,
)
from radjax_student.objectives import (
    SPARSE_CROSS_ENTROPY_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry, SgdOptimizer
from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry

pytestmark = pytest.mark.jax

_TOKENS = (1, 7, 3, 5)
_TARGETS = (7, 3, 5, 1)
GRADIENT_BOUNDARY = (
    "Gradients differentiate through multiple token positions within one fixture "
    "sequence.",
    "Carry returned from one learning step can seed a later learning step.",
    "Carry crossing separate learning-step boundaries is stop-gradient state.",
    "Full cross-step BPTT, truncated-BPTT scheduling, and long-context recurrent "
    "training are not proven.",
)


def _batch(tokens: tuple[int, ...] = _TOKENS) -> LearningBatch:
    return LearningBatch(
        "p45-rwkv7",
        inputs={"token_ids": [list(tokens)]},
        targets={"token_ids": [list(_TARGETS)]},
    )


def _assembled(compilation_policy: str):
    architecture_registry = ArchitectureRegistry()
    register_rwkv7_reference(architecture_registry)
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(SgdOptimizer())
    request = JaxLearningAssemblyRequest(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        architecture_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        architecture_config=reference_architecture_config(),
        objective_identity=SPARSE_CROSS_ENTROPY_IDENTITY,
        objective_config=ObjectiveConfig(
            SPARSE_CROSS_ENTROPY_IDENTITY, {"reduction": "mean"}
        ),
        optimizer_id="sgd.v1",
        optimizer_version=1,
        optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.05),
        runtime_backend_id="jax",
        runtime_implementation_version="p2.9",
        runtime_config=RuntimeConfig(
            backend_id="jax",
            platform_preference="cpu",
            precision_policy="float32",
            placement_policy="single_device",
            compilation_policy=compilation_policy,
            distributed_policy="disabled",
            fallback_policy="disallowed",
            seed=17,
        ),
        root_seed=17,
        learning_state=LearningState(
            "p45-rwkv7",
            active_update_scope=UpdateScope("whole_student"),
            active_objective_scope=ObjectiveScope(),
        ),
    )
    return assemble_jax_learning_lifecycle(
        request,
        registries=JaxLearningAssemblyRegistries(
            architecture_registry,
            build_default_objective_registry(),
            optimizer_registry,
            build_default_runtime_registry(),
        ),
    )


def _execute(assembled, batch: LearningBatch):
    lifecycle = assembled.loop_executor.lifecycle
    return assembled.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        parameters=lifecycle.parameters,
        objective=lifecycle.objective_selection,
        batch=batch,
    )


def _tree_allclose(first: Any, second: Any) -> bool:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    return first_tree == second_tree and all(
        bool(jnp.allclose(a, b, rtol=1e-5, atol=2e-5))
        for a, b in zip(first_leaves, second_leaves, strict=True)
    )


def _tree_changed(first: Any, second: Any) -> bool:
    return any(
        not bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(first),
            jax.tree_util.tree_leaves(second),
            strict=True,
        )
    )


def _all_finite(value: Any) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(value)
    )


def test_rwkv_batch_validation_accepts_only_the_finite_json_tiny_domain() -> None:
    plugin = RWKV7ReferencePlugin()
    config = reference_architecture_config()

    assert plugin.validate_batch(_batch(), config).status == "pass"
    for malformed in (
        ((1, 7, 3),),
        ((1, 7, 3, 16),),
        ((1, 7, 3, True),),
    ):
        batch = LearningBatch(
            "p45-rwkv7",
            inputs={"token_ids": [list(malformed[0])]},
            targets={"token_ids": [list(_TARGETS)]},
        )
        validation = plugin.validate_batch(batch, config)
        assert validation.status == "fail"
        assert validation.blockers[0].code == "architecture_batch_incompatible"


def test_eager_and_jit_registered_lifecycles_agree_and_advance_state() -> None:
    eager = _assembled("eager")
    compiled = _assembled("jit")
    eager_before = eager.lifecycle
    compiled_before = compiled.lifecycle

    eager_execution = _execute(eager, _batch())
    compiled_execution = _execute(compiled, _batch())
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
        assert _all_finite(execution.gradients)
        assert execution.result.changed_parameter_paths
        assert _tree_changed(before.parameters, after.parameters)
        assert _tree_changed(before.architecture_carry, after.architecture_carry)
        assert _all_finite(after.architecture_carry)
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
    assert _tree_allclose(eager_after.parameters, compiled_after.parameters)
    assert _tree_allclose(
        eager_after.architecture_carry, compiled_after.architecture_carry
    )
    assert eager_execution.result.loss.loss == pytest.approx(
        compiled_execution.result.loss.loss, rel=1e-5, abs=2e-5
    )

    later_before = eager.loop_executor.lifecycle
    assert _tree_allclose(
        later_before.architecture_carry, eager_after.architecture_carry
    )
    later = _execute(eager, _batch((5, 3, 7, 1)))
    later_after = eager.loop_executor.lifecycle
    assert later.result.status == "pass"
    assert later_after.learning_state.global_step == 2
    assert later_after.optimizer_state.envelope.step == 2
    assert later.runtime_result.output_metadata["rng_bridge"]["global_step"] == 1
    assert later.runtime_result.output_metadata["rng_bridge"]["invocation_index"] == 1


def test_within_sequence_gradients_and_cross_step_carry_boundary() -> None:
    assembled = _assembled("eager")
    lifecycle = assembled.lifecycle
    loss_fn = build_registered_jax_loss_fn(
        architecture=lifecycle.architecture,
        objective_selection=lifecycle.objective_selection,
        objective_config=lifecycle.objective_config,
        objective_descriptor=lifecycle.objective_descriptor,
        resolved_selection=lifecycle.resolved_objective_selection,
    )
    materializer = FiniteJsonJaxBatchMaterializer()
    base_batch = materializer.materialize(_batch())
    value_and_grad = build_value_and_grad_fn(loss_fn)

    def gradient_for(tokens: tuple[int, ...]):
        batch = materializer.materialize(_batch(tokens))
        (_, _), gradients = value_and_grad(
            lifecycle.parameters,
            lifecycle.architecture_carry,
            batch,
            None,
        )
        return gradients

    base = gradient_for(_TOKENS)
    changed_first = gradient_for((2, 7, 3, 5))
    changed_third = gradient_for((1, 7, 4, 5))
    assert _all_finite(base)
    assert _tree_changed(base, changed_first)
    assert _tree_changed(base, changed_third)

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
