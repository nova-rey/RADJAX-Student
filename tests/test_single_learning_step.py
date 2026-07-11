from __future__ import annotations

from dataclasses import dataclass

from radjax_student.architecture import ArchitectureConfig
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.learning import LearningBatch, LearningState, UpdateScope
from radjax_student.optimizers import (
    OptimizerConfig,
    OptimizerInitRequest,
    SgdOptimizer,
)
from radjax_student.steps import learning_step


@dataclass(frozen=True)
class LinearObjective:
    def evaluate(self, parameters, batch):
        x = float(batch.inputs["token_ids"]["x"])
        target = float(batch.targets["target"]["y"])
        prediction = parameters["trunk.weight"] * x + parameters["head.weight"]
        error = prediction - target
        return error * error, {
            "head.weight": 2.0 * error,
            "trunk.bias": 0.0,
            "trunk.weight": 2.0 * error * x,
        }


def _run(scope: UpdateScope | None = None):
    architecture = FakeArchitecturePlugin()
    config = ArchitectureConfig(
        architecture_id=architecture.architecture_id, sequence_length=4
    )
    batch = LearningBatch(
        batch_id="synthetic",
        inputs={"token_ids": {"rank": 2, "sequence_length": 1, "x": 1.0}},
        targets={"target": {"y": 3.0}},
    )
    optimizer = SgdOptimizer()
    optimizer_config = OptimizerConfig(
        optimizer_id=optimizer.optimizer_id, learning_rate=0.1
    )
    catalog = architecture.describe_parameters()
    scope = scope or UpdateScope()
    selection = architecture.resolve_update_scope(scope, catalog)
    optimizer_state = optimizer.initialize_state(
        OptimizerInitRequest(optimizer_config, catalog, selection)
    ).optimizer_state
    return learning_step(
        batch=batch,
        architecture=architecture,
        architecture_config=config,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        learning_state=LearningState(run_id="single", active_update_scope=scope),
        parameters={"head.weight": 0.0, "trunk.bias": 0.0, "trunk.weight": 0.0},
        objective=LinearObjective(),
    )


def test_single_step_changes_parameters_and_advances_state():
    execution = _run()
    assert execution.result.ok and execution.result.loss.loss == 9.0
    assert execution.learning_state.global_step == execution.optimizer_state.step == 1
    assert execution.parameters["trunk.weight"] != 0.0
    assert {metric.name for metric in execution.result.metrics} >= {
        "loss",
        "gradient_norm",
        "parameter_norm",
        "learning_rate",
        "changed_parameter_count",
        "unchanged_parameter_count",
        "step_time",
    }


def test_single_step_respects_named_region_and_replays_deterministically():
    scope = UpdateScope(kind="named_region", region_id="trunk")
    partial, replay = _run(scope), _run(scope)
    assert partial.parameters == replay.parameters
    assert partial.parameters["head.weight"] == 0.0
    assert partial.parameters["trunk.weight"] != 0.0
    assert (
        partial.optimizer_state.backend_state["per_parameter_steps"]["head.weight"] == 0
    )
