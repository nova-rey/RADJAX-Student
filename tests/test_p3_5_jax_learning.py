from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.architecture import (  # noqa: E402
    ArchitectureContractError,
    ForwardResult,
    JaxArchitectureExecution,
)
from radjax_student.learning import ObjectiveScope  # noqa: E402
from radjax_student.learning.jax_core import (  # noqa: E402
    JaxBatch,
    JaxLossAuxiliary,
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
from radjax_student.steps.jax_step import execute_jax_learning_step  # noqa: E402

pytestmark = pytest.mark.jax


@dataclass
class LinearJaxArchitecture:
    stochastic: bool = False
    scopes: list[ObjectiveScope] = field(default_factory=list)

    def apply_jax(
        self,
        parameters: Any,
        architecture_state: Any,
        batch: JaxBatch,
        *,
        objective_scope: ObjectiveScope,
        training: bool,
        rng_key: Any | None,
    ) -> ForwardResult:
        assert isinstance(objective_scope, ObjectiveScope)
        self.scopes.append(objective_scope)
        x = batch.inputs["x"][:, None]
        prediction = jnp.dot(x, parameters["weight"][None, :]) + parameters["bias"]
        if self.stochastic and rng_key is not None:
            prediction = prediction + jax.random.uniform(rng_key, prediction.shape)
        return ForwardResult(
            outputs=prediction,
            surface_values={"hidden": prediction + 1.0},
            updated_architecture_carry={"counter": architecture_state["counter"] + 1},
            architecture_metrics={"counter": architecture_state["counter"]},
        )


class MeanSquaredError:
    def __init__(self) -> None:
        self.values: list[Any] = []

    def evaluate(self, surface, targets, weights, objective_config):
        del weights
        self.values.append(surface)
        errors = surface - targets["y"]
        loss = jnp.mean(errors**2)
        if objective_config.reduction == "sum":
            loss = jnp.sum(errors**2)
        return loss, {"mse": loss}


def _setup(*, stochastic: bool = False, scope: ObjectiveScope | None = None):
    architecture = LinearJaxArchitecture(stochastic=stochastic)
    objective = MeanSquaredError()
    batch = JaxBatch(
        inputs={"x": jnp.asarray((-1.0, 0.0, 1.0))},
        targets={"y": jnp.asarray(((-1.0,), (1.0,), (3.0,)))},
    )
    parameters = {"weight": jnp.asarray((0.0,)), "bias": jnp.asarray((0.0,))}
    carry = {"counter": jnp.asarray(0.0)}
    config = JaxObjectiveConfig("mse", scope or ObjectiveScope())
    return architecture, objective, batch, parameters, carry, config


def _runtime(mode: str = "jit"):
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
        request_id=f"p35-jax-{mode}",
        function_id="p35.linear_loss",
        mode=mode,
        compilation_options=CompilationOptions(mode=mode, synchronize_results=True),
    )
    return context, backend, request


def _loss_and_grad():
    architecture, objective, batch, parameters, carry, config = _setup()
    return (
        architecture,
        objective,
        batch,
        parameters,
        carry,
        config,
        build_value_and_grad_fn(build_jax_loss_fn(architecture, objective)),
    )


def test_jax_objective_receives_surface_not_parameters():
    architecture, objective, batch, parameters, carry, config, value_and_grad = (
        _loss_and_grad()
    )
    value_and_grad(parameters, carry, batch, config, None)
    assert objective.values and objective.values[0] is not parameters
    assert objective.values[0].shape == (3, 1)
    assert isinstance(architecture, JaxArchitectureExecution)


def test_jax_architecture_receives_objective_scope_object():
    architecture, _, batch, parameters, carry, config, value_and_grad = _loss_and_grad()
    value_and_grad(parameters, carry, batch, config, None)
    assert architecture.scopes[-1] == config.objective_scope


def test_jax_final_output_surface_resolves():
    architecture, objective, batch, parameters, carry, config, _ = _loss_and_grad()
    loss, auxiliary = build_jax_loss_fn(architecture, objective)(
        parameters, carry, batch, config, None
    )
    assert float(loss) == pytest.approx(11.0 / 3.0)
    assert isinstance(auxiliary, JaxLossAuxiliary)
    assert auxiliary.selected_surface.shape == (3, 1)


def test_jax_intermediate_surface_resolves():
    architecture, objective, batch, parameters, carry, _, _ = _loss_and_grad()
    config = JaxObjectiveConfig("mse", ObjectiveScope("intermediate_surface", "hidden"))
    _, auxiliary = build_jax_loss_fn(architecture, objective)(
        parameters, carry, batch, config, None
    )
    assert float(auxiliary.selected_surface[1, 0]) == pytest.approx(1.0)


def test_jax_missing_surface_rejected():
    architecture, objective, batch, parameters, carry, _, _ = _loss_and_grad()
    config = JaxObjectiveConfig("mse", ObjectiveScope("intermediate_surface", "none"))
    with pytest.raises(ArchitectureContractError, match="unavailable"):
        build_jax_loss_fn(architecture, objective)(
            parameters, carry, batch, config, None
        )


def test_jax_loss_gradients_match_expected_linear_solution():
    _, _, batch, parameters, carry, config, value_and_grad = _loss_and_grad()
    (loss, _), gradients = value_and_grad(parameters, carry, batch, config, None)
    assert float(loss) == pytest.approx(11.0 / 3.0)
    assert float(gradients["weight"][0]) == pytest.approx(-8.0 / 3.0)
    assert float(gradients["bias"][0]) == pytest.approx(-2.0)


def test_jax_gradients_only_target_parameters():
    _, _, batch, parameters, carry, config, value_and_grad = _loss_and_grad()
    (_, auxiliary), gradients = value_and_grad(parameters, carry, batch, config, None)
    assert set(gradients) == set(parameters)
    assert set(auxiliary.updated_architecture_carry) == {"counter"}


def test_jax_input_carry_is_stop_gradient_isolated():
    architecture, objective, batch, parameters, _, config, _ = _loss_and_grad()

    def loss(state):
        return build_jax_loss_fn(architecture, objective)(
            parameters, {"counter": state}, batch, config, None
        )[0]

    assert float(jax.grad(loss)(jnp.asarray(2.0))) == 0.0


def test_jax_output_carry_is_stop_gradient_isolated():
    _, _, batch, parameters, carry, config, value_and_grad = _loss_and_grad()
    (_, auxiliary), _ = value_and_grad(parameters, carry, batch, config, None)
    assert (
        float(
            jax.grad(lambda value: value)(
                auxiliary.updated_architecture_carry["counter"]
            )
        )
        == 1.0
    )
    assert float(auxiliary.updated_architecture_carry["counter"]) == 1.0


def test_jax_functional_carry_transition():
    architecture, objective, batch, parameters, carry, config, _ = _loss_and_grad()
    _, auxiliary = build_jax_loss_fn(architecture, objective)(
        parameters, carry, batch, config, None
    )
    assert float(carry["counter"]) == 0.0
    assert float(auxiliary.updated_architecture_carry["counter"]) == 1.0


def test_jax_same_rng_key_replays_exactly():
    architecture, objective, batch, parameters, carry, config = _setup(stochastic=True)
    loss_fn = build_jax_loss_fn(architecture, objective)
    first = loss_fn(parameters, carry, batch, config, jax.random.key(4))[0]
    second = loss_fn(parameters, carry, batch, config, jax.random.key(4))[0]
    assert float(first) == float(second)


def test_jax_different_rng_key_changes_stochastic_fixture():
    architecture, objective, batch, parameters, carry, config = _setup(stochastic=True)
    loss_fn = build_jax_loss_fn(architecture, objective)
    first = loss_fn(parameters, carry, batch, config, jax.random.key(4))[0]
    second = loss_fn(parameters, carry, batch, config, jax.random.key(5))[0]
    assert float(first) != float(second)


def test_jax_nonfinite_loss_rejected():
    with pytest.raises(ValueError, match="loss"):
        validate_finite_loss_and_gradients(
            jnp.asarray(float("nan")), {"x": jnp.asarray(1.0)}
        )


def test_jax_nonfinite_gradient_rejected():
    with pytest.raises(ValueError, match="gradients"):
        validate_finite_loss_and_gradients(
            jnp.asarray(1.0), {"x": jnp.asarray(float("inf"))}
        )


def test_jax_scoped_update_preserves_excluded_parameter():
    parameters = {"weight": jnp.asarray((0.0,)), "bias": jnp.asarray((0.0,))}
    updated = apply_scoped_gradient_update(
        parameters,
        {"weight": jnp.asarray((-2.0,)), "bias": jnp.asarray((-2.0,))},
        {"weight": True, "bias": False},
        0.1,
    )
    assert float(updated["weight"][0]) == pytest.approx(0.2)
    assert float(updated["bias"][0]) == 0.0


def test_jax_multi_step_loss_reduction():
    architecture, objective, batch, parameters, carry, config, value_and_grad = (
        _loss_and_grad()
    )
    loss_fn = build_jax_loss_fn(architecture, objective)
    first_loss = loss_fn(parameters, carry, batch, config, None)[0]
    for _ in range(12):
        (_, auxiliary), gradients = value_and_grad(
            parameters, carry, batch, config, None
        )
        parameters = apply_scoped_gradient_update(
            parameters, gradients, {"weight": True, "bias": True}, 0.15
        )
        carry = auxiliary.updated_architecture_carry
    assert float(loss_fn(parameters, carry, batch, config, None)[0]) < float(first_loss)


def test_jax_multi_step_replay_is_exact():
    def run():
        architecture, objective, batch, parameters, carry, config, value_and_grad = (
            _loss_and_grad()
        )
        for _ in range(4):
            (_, auxiliary), gradients = value_and_grad(
                parameters, carry, batch, config, None
            )
            parameters = apply_scoped_gradient_update(
                parameters, gradients, {"weight": True, "bias": True}, 0.1
            )
            carry = auxiliary.updated_architecture_carry
        return parameters, carry

    first, first_carry = run()
    second, second_carry = run()
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, first, second)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, first_carry, second_carry)
    )


def test_jax_eager_and_runtime_jit_match():
    architecture, objective, batch, parameters, carry, config, _ = _loss_and_grad()
    loss_fn = build_jax_loss_fn(architecture, objective)
    output, receipt = execute_function(
        context=_runtime()[0],
        function=loss_fn,
        request=_runtime()[2],
        backend=_runtime()[1],
        args=(parameters, carry, batch, config, None),
    )
    assert receipt.status == "pass" and receipt.compiled
    assert float(output[0]) == pytest.approx(
        float(loss_fn(parameters, carry, batch, config, None)[0])
    )


def test_jax_jit_path_uses_runtime_execution_boundary(monkeypatch):
    architecture, objective, batch, parameters, carry, config, _ = _loss_and_grad()
    import radjax_student.steps.jax_step as jax_step

    calls = []
    original = jax_step.execute_function

    def observed(**kwargs):
        calls.append(kwargs["request"].mode)
        return original(**kwargs)

    monkeypatch.setattr(jax_step, "execute_function", observed)
    context, backend, request = _runtime()
    execution = execute_jax_learning_step(
        architecture=architecture,
        objective=objective,
        parameters=parameters,
        architecture_carry=carry,
        batch=batch,
        objective_config=config,
        rng_key=None,
        selection_mask={"weight": True, "bias": True},
        learning_rate=0.1,
        runtime_context=context,
        runtime_backend=backend,
        execution_request=request,
    )
    assert calls == ["jit"] and execution.runtime_result.status == "pass"


def test_jax_core_does_not_call_jax_jit():
    tree = ast.parse(Path("src/radjax_student/learning/jax_core.py").read_text())
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "jit"
        for node in ast.walk(tree)
    )


def test_jax_architecture_does_not_select_devices():
    tree = ast.parse(Path(__file__).read_text())
    selected = {"device_put", "devices", "local_devices"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in selected
        for node in ast.walk(tree)
    )


def test_jaxpr_contains_one_architecture_forward_dot():
    architecture, objective, batch, parameters, carry, config, _ = _loss_and_grad()
    jaxpr = jax.make_jaxpr(build_jax_loss_fn(architecture, objective))(
        parameters, carry, batch, config, None
    )
    primitives = [equation.primitive.name for equation in jaxpr.jaxpr.eqns]
    assert primitives.count("dot_general") == 1


def test_missing_jax_capability_fails_explicitly():
    class Missing:
        marker = "missing"

    with pytest.raises(TypeError, match="JAX"):
        build_jax_loss_fn(Missing(), MeanSquaredError())


def test_missing_jax_capability_does_not_use_legacy_fallback(monkeypatch):
    class Missing:
        marker = "missing"

    monkeypatch.setattr(
        "radjax_student.legacy.scalar_learning.legacy_scalar_learning_step",
        lambda **_: pytest.fail("legacy fallback used"),
    )
    with pytest.raises(TypeError, match="JAX"):
        build_jax_loss_fn(Missing(), MeanSquaredError())


def test_jax_correctness_path_imports_no_numpy():
    paths = (
        Path("src/radjax_student/learning/jax_core.py"),
        Path("src/radjax_student/steps/jax_step.py"),
    )
    assert all("import numpy" not in path.read_text() for path in paths)
