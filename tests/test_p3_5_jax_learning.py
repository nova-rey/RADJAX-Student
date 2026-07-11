from __future__ import annotations

from dataclasses import dataclass

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.architecture import (  # noqa: E402
    ForwardResult,
    JaxArchitectureExecution,
)
from radjax_student.learning.jax_core import (  # noqa: E402
    JaxBatch,
    JaxObjectiveConfig,
    apply_scoped_gradient_update,
    build_jax_loss_fn,
    build_value_and_grad_fn,
    validate_finite_loss_and_gradients,
)
from radjax_student.runtime import (  # noqa: E402
    CompilationOptions,
    DeviceInventory,
    ExecutionContext,
    ExecutionRequest,
    JaxRuntimeBackend,
    RuntimeEnvironment,
    execute_function,
)

pytestmark = pytest.mark.jax


@dataclass(frozen=True)
class LinearJaxArchitecture:
    architecture_id: str = "p35.linear.v1"

    def apply_jax(
        self,
        parameters,
        architecture_state,
        batch,
        *,
        objective_scope,
        training,
        rng_key,
    ):
        del objective_scope, training, rng_key
        x = batch.inputs["x"]
        output = parameters["weight"] * x + parameters["bias"]
        return ForwardResult(
            outputs=output,
            surface_values={"final_output": output},
            updated_runtime_state=architecture_state,
        )


class MeanSquaredError:
    def evaluate(self, surface, targets, weights, objective_config):
        del weights
        errors = surface - targets["y"]
        loss = (
            jnp.mean(errors**2)
            if objective_config.reduction == "mean"
            else jnp.sum(errors**2)
        )
        return loss, {"mse": loss}


def _loss_setup():
    architecture = LinearJaxArchitecture()
    objective = MeanSquaredError()
    loss_fn = build_jax_loss_fn(architecture, objective)
    batch = JaxBatch(
        inputs={"x": jnp.asarray((-1.0, 0.0, 1.0))},
        targets={"y": jnp.asarray((-1.0, 1.0, 3.0))},
    )
    config = JaxObjectiveConfig("mse", surface_id="final_output")
    parameters = {"weight": jnp.asarray(0.0), "bias": jnp.asarray(0.0)}
    state = {"carry": jnp.asarray(0.0)}
    return loss_fn, batch, config, parameters, state


def test_jax_linear_loss_uses_forward_surface_and_autodiff():
    loss_fn, batch, config, parameters, state = _loss_setup()
    (loss, auxiliary), gradients = build_value_and_grad_fn(loss_fn)(
        parameters, state, batch, config, None
    )

    validate_finite_loss_and_gradients(loss, gradients)
    assert float(loss) == pytest.approx(11.0 / 3.0)
    assert set(gradients) == set(parameters)
    assert auxiliary["updated_runtime_state"] == state


def test_jax_eager_and_runtime_jit_match():
    loss_fn, batch, config, parameters, state = _loss_setup()
    eager = loss_fn(parameters, state, batch, config, None)
    backend = JaxRuntimeBackend()
    context = ExecutionContext(
        backend_id="jax",
        environment=RuntimeEnvironment(
            python_version="3.11",
            jax_available=True,
            process_count=1,
            process_index=0,
            local_device_count=1,
            global_device_count=1,
            distributed_initialized=False,
        ),
        device_inventory=DeviceInventory(
            process_count=1, local_device_count=1, global_device_count=1
        ),
        capabilities=backend.capability_profile(),
        root_seed=0,
        runtime_id="p35-jax-test",
    )
    request = ExecutionRequest(
        request_id="p35-jax-jit",
        function_id="p35.linear_loss",
        mode="jit",
        compilation_options=CompilationOptions(
            mode="jit",
            synchronize_results=True,
        ),
    )
    output, result = execute_function(
        context=context,
        function=loss_fn,
        request=request,
        backend=backend,
        args=(parameters, state, batch, config, None),
    )

    assert result.status == "pass"
    assert output is not None
    assert float(output[0]) == pytest.approx(float(eager[0]))
    assert result.compiled is True


def test_jax_scoped_update_preserves_excluded_parameter():
    parameters = {"weight": jnp.asarray(0.0), "bias": jnp.asarray(0.0)}
    gradients = {"weight": jnp.asarray(-2.0), "bias": jnp.asarray(-2.0)}
    updated = apply_scoped_gradient_update(
        parameters, gradients, {"weight": True, "bias": False}, 0.1
    )

    assert float(updated["weight"]) == pytest.approx(0.2)
    assert float(updated["bias"]) == 0.0


def test_jax_execution_protocol_is_one_architecture_capability():
    assert isinstance(LinearJaxArchitecture(), JaxArchitectureExecution)
