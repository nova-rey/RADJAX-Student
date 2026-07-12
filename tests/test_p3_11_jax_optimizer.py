"""P3.11.4 proves JAX SGD remains behind the existing optimizer identity."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.contracts import (  # noqa: E402
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.optimizers import (  # noqa: E402
    JaxOptimizerExecution,
    OptimizerConfig,
    OptimizerContractError,
    OptimizerState,
    SgdOptimizer,
    advanced_jax_optimizer_state,
    require_finite_jax_gradients,
    validate_jax_optimizer_state,
)

pytestmark = pytest.mark.jax


def _layout() -> ParameterTreeLayout:
    return ParameterTreeLayout(
        "test.architecture.v1",
        (
            ParameterTreeLayoutEntry(
                "head.weight", ("head", "weight"), (1,), "float32", "output_head"
            ),
            ParameterTreeLayoutEntry(
                "trunk.weight",
                ("trunk", "weight"),
                (1,),
                "float32",
                "recurrent_block",
            ),
        ),
    )


def _state(optimizer: SgdOptimizer, layout: ParameterTreeLayout):
    return optimizer.initialize_jax_state(
        config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1),
        parameter_layout=layout,
        optimizer_state=OptimizerState(optimizer.optimizer_id, layout.logical_paths),
    )


def test_jax_sgd_uses_existing_optimizer_identity_and_layout_descriptor():
    optimizer = SgdOptimizer()
    layout = _layout()
    state = _state(optimizer, layout)
    assert isinstance(optimizer, JaxOptimizerExecution)
    validate_jax_optimizer_state(
        state,
        optimizer_id=optimizer.optimizer_id,
        parameter_layout=layout,
        descriptor=optimizer.jax_state_descriptor(layout),
    )
    assert state.descriptor.state_keypaths == (
        ("per_parameter_steps", "head", "weight"),
        ("per_parameter_steps", "trunk", "weight"),
        ("step",),
    )


def test_jax_sgd_honors_schedule_and_preserves_excluded_parameter_state():
    optimizer = SgdOptimizer()
    layout = _layout()
    state = _state(optimizer, layout)
    parameters = {
        "head": {"weight": jnp.asarray((4.0,))},
        "trunk": {"weight": jnp.asarray((1.0,))},
    }
    gradients = {
        "head": {"weight": jnp.asarray((2.0,))},
        "trunk": {"weight": jnp.asarray((3.0,))},
    }
    mask = layout.update_mask(parameters, ("trunk.weight",))
    updated, arrays, metrics = optimizer.apply_jax_updates(
        parameters=parameters,
        gradients=gradients,
        optimizer_array_state=state.arrays,
        update_mask=mask,
        config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1),
        schedule_values={"learning_rate": 0.25},
    )
    require_finite_jax_gradients(metrics)
    assert float(updated["trunk"]["weight"][0]) == pytest.approx(0.25)
    assert float(updated["head"]["weight"][0]) == 4.0
    assert int(arrays["per_parameter_steps"]["trunk"]["weight"]) == 1
    assert int(arrays["per_parameter_steps"]["head"]["weight"]) == 0
    advanced = advanced_jax_optimizer_state(state, arrays)
    assert advanced.envelope.step == 1 and int(advanced.arrays["step"]) == 1


def test_jax_sgd_rejects_nonfinite_gradients_and_descriptor_mismatch():
    optimizer = SgdOptimizer()
    layout = _layout()
    state = _state(optimizer, layout)
    parameters = {
        "head": {"weight": jnp.asarray((1.0,))},
        "trunk": {"weight": jnp.asarray((1.0,))},
    }
    _, _, metrics = optimizer.apply_jax_updates(
        parameters=parameters,
        gradients={
            "head": {"weight": jnp.asarray((1.0,))},
            "trunk": {"weight": jnp.asarray((jnp.inf,))},
        },
        optimizer_array_state=state.arrays,
        update_mask=layout.update_mask(parameters, layout.logical_paths),
        config=OptimizerConfig(optimizer.optimizer_id),
        schedule_values={},
    )
    with pytest.raises(OptimizerContractError, match="finite"):
        require_finite_jax_gradients(metrics)
    with pytest.raises(OptimizerContractError, match="layout digest"):
        validate_jax_optimizer_state(
            state,
            optimizer_id=optimizer.optimizer_id,
            parameter_layout=ParameterTreeLayout(
                "test.architecture.v1",
                (
                    ParameterTreeLayoutEntry(
                        "other", ("other",), (1,), "float32", "other"
                    ),
                ),
            ),
            descriptor=optimizer.jax_state_descriptor(layout),
        )
