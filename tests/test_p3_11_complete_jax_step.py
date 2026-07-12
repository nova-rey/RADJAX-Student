"""P3.11.5 proves runtime executes the complete JAX update, not only gradients."""

from __future__ import annotations

from typing import Any

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.architecture import (  # noqa: E402
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureMetadata,
    BatchValidationResult,
    ForwardResult,
    IntermediateSurfaceDescriptor,
    NamedRegion,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.architecture.testing import (  # noqa: E402
    FAKE_ARCHITECTURE_CAPABILITIES,
    FakeArchitecturePlugin,
)
from radjax_student.contracts import (  # noqa: E402
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.learning import (  # noqa: E402
    LearningBatch,
    LearningState,
    ObjectiveScope,
    UpdateScope,
)
from radjax_student.learning.jax_core import JaxBatch, JaxObjectiveConfig  # noqa: E402
from radjax_student.optimizers import (  # noqa: E402
    OptimizerConfig,
    OptimizerState,
    SgdOptimizer,
)
from radjax_student.runtime import (  # noqa: E402
    CompilationOptions,
    DeviceInventory,
    ExecutionContext,
    ExecutionRequest,
    JaxRuntimeBackend,
    RuntimeEnvironment,
)
from radjax_student.steps.jax_step import execute_jax_learning_step  # noqa: E402

pytestmark = pytest.mark.jax


class LinearPlugin(FakeArchitecturePlugin):
    def validate_batch(self, batch, config):
        self.validate_config(config)
        return BatchValidationResult("pass")

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            self.architecture_version,
            (*FAKE_ARCHITECTURE_CAPABILITIES, "architecture.jax_execution_v1"),
        )

    def describe_parameters(self, parameters: object | None = None) -> ParameterCatalog:
        del parameters
        return ParameterCatalog(
            self.architecture_id,
            (
                ParameterDescriptor(
                    "trunk.weight", (1,), "float32", "recurrent_block", ("trunk",)
                ),
                ParameterDescriptor(
                    "head.bias", (1,), "float32", "output_head", ("head",)
                ),
            ),
        )

    def architecture_metadata(self) -> ArchitectureMetadata:
        catalog = self.describe_parameters()
        return ArchitectureMetadata(
            self.architecture_id,
            catalog,
            self.capability_profile(),
            named_regions=(
                NamedRegion("trunk", ("trunk.weight",)),
                NamedRegion("head", ("head.bias",)),
            ),
            objective_surfaces=(
                IntermediateSurfaceDescriptor(
                    "final_output", "logits", available_in_training=True
                ),
            ),
        )

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
        del objective_scope, training, rng_key
        output = (
            batch.inputs["x"][:, None] * parameters["trunk"]["weight"]
            + parameters["head"]["bias"]
        )
        return ForwardResult(
            outputs=output,
            updated_architecture_carry={"count": architecture_state["count"] + 1},
            architecture_metrics={"carry_count": architecture_state["count"]},
        )


class MeanSquaredError:
    seen_objective_ids = []

    def evaluate(self, surface, targets, weights, objective_config):
        del weights
        self.seen_objective_ids.append(objective_config.objective_id)
        loss = jnp.mean((surface - targets["y"]) ** 2)
        return loss, {"mse": loss}


def _layout() -> ParameterTreeLayout:
    return ParameterTreeLayout(
        "test.architecture.v1",
        (
            ParameterTreeLayoutEntry(
                "trunk.weight",
                ("trunk", "weight"),
                (1,),
                "float32",
                "recurrent_block",
                ("trunk",),
            ),
            ParameterTreeLayoutEntry(
                "head.bias", ("head", "bias"), (1,), "float32", "output_head", ("head",)
            ),
        ),
    )


def _runtime(mode: str):
    backend = JaxRuntimeBackend()
    context = ExecutionContext(
        backend_id="jax",
        environment=RuntimeEnvironment(
            python_version="3.11",
            jax_available=True,
            local_device_count=1,
            global_device_count=1,
        ),
        device_inventory=DeviceInventory(local_device_count=1, global_device_count=1),
        capabilities=backend.capability_profile(),
        root_seed=0,
        runtime_id="p311-step",
        metadata={
            "selected_device_id": "cpu:0",
            "placement_policy": "single_device",
            "precision_policy": "float32",
        },
    )
    request = ExecutionRequest(
        request_id=f"p311-step-{mode}",
        function_id="p311.complete_step",
        mode=mode,
        compilation_options=CompilationOptions(mode=mode, synchronize_results=True),
    )
    return context, backend, request


@pytest.mark.parametrize("mode", ("eager", "jit"))
def test_runtime_receipt_covers_full_scoped_jax_update(mode: str):
    architecture = LinearPlugin()
    layout = _layout()
    optimizer = SgdOptimizer()
    optimizer_state = optimizer.initialize_jax_state(
        config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1),
        parameter_layout=layout,
        optimizer_state=OptimizerState(optimizer.optimizer_id, layout.logical_paths),
    )
    parameters = {
        "trunk": {"weight": jnp.asarray((0.0,))},
        "head": {"bias": jnp.asarray((0.0,))},
    }
    context, backend, request = _runtime(mode)
    execution = execute_jax_learning_step(
        architecture=architecture,
        architecture_config=ArchitectureConfig(architecture.architecture_id),
        objective=MeanSquaredError(),
        optimizer=optimizer,
        optimizer_config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1),
        optimizer_state=optimizer_state,
        learning_state=LearningState(
            "p311",
            active_update_scope=UpdateScope("named_region", "trunk"),
            active_objective_scope=ObjectiveScope(),
        ),
        parameters=parameters,
        architecture_carry={"count": jnp.asarray(0)},
        batch=JaxBatch(
            {"x": jnp.asarray((-1.0, 0.0, 1.0))},
            {"y": jnp.asarray(((-1.0,), (1.0,), (3.0,)))},
        ),
        learning_batch=LearningBatch(
            "linear",
            inputs={"x": [-1.0, 0.0, 1.0]},
            targets={"y": [[-1.0], [1.0], [3.0]]},
        ),
        objective_config=JaxObjectiveConfig("linear.mse.v1"),
        parameter_layout=layout,
        runtime_context=context,
        runtime_backend=backend,
        execution_request=request,
    )
    assert execution.runtime_result.status == "pass"
    assert (
        execution.runtime_result.output_metadata["input_preparation"][
            "precision_policy"
        ]
        == "float32"
    )
    assert (
        execution.runtime_result.output_metadata["input_preparation"][
            "selected_device_id"
        ]
        == "cpu:0"
    )
    assert execution.runtime_result.output_metadata["rng_bridge"]["stream"] == "dropout"
    assert "linear.mse.v1" in MeanSquaredError.seen_objective_ids
    assert execution.result.changed_parameter_paths == ("trunk.weight",)
    assert execution.result.unchanged_parameter_paths == ("head.bias",)
    assert float(execution.parameters["trunk"]["weight"][0]) != 0.0
    assert float(execution.parameters["head"]["bias"][0]) == 0.0
    assert (
        execution.learning_state.global_step
        == execution.optimizer_state.envelope.step
        == 1
    )
    assert int(execution.architecture_carry["count"]) == 1
