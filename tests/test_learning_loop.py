from radjax_student.architecture import ArchitectureConfig
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.learning import LearningBatch, LearningState
from radjax_student.optimizers import (
    OptimizerConfig,
    OptimizerInitRequest,
    SgdOptimizer,
)
from radjax_student.steps import (
    LearningLoopConfig,
    SyntheticBatchSource,
    run_learning_loop,
)
from tests.test_single_learning_step import LinearObjective


def test_loop_stops_exactly_and_restores_source_position():
    architecture = FakeArchitecturePlugin()
    catalog = architecture.describe_parameters()
    optimizer = SgdOptimizer()
    config = OptimizerConfig(optimizer_id="sgd.v1", learning_rate=0.1)
    state = optimizer.initialize_state(
        OptimizerInitRequest(
            config,
            catalog,
            architecture.resolve_update_scope(
                LearningState(run_id="r").active_update_scope, catalog
            ),
        )
    ).optimizer_state
    batch = LearningBatch(
        batch_id="b",
        inputs={"token_ids": {"rank": 2, "sequence_length": 1, "x": 1.0}},
        targets={"target": {"y": 3.0}},
    )
    source = SyntheticBatchSource((batch, batch, batch))
    result = run_learning_loop(
        config=LearningLoopConfig(max_steps=2),
        architecture=architecture,
        architecture_config=ArchitectureConfig(
            architecture_id=architecture.architecture_id, sequence_length=4
        ),
        optimizer=optimizer,
        optimizer_config=config,
        optimizer_state=state,
        learning_state=LearningState(run_id="r"),
        parameters={"head.weight": 0.0, "trunk.bias": 0.0, "trunk.weight": 0.0},
        objective=LinearObjective(),
        batch_source=source,
    )
    assert (
        result.stop_reason == "max_steps"
        and result.steps_completed == 2
        and source.position == 2
    )
    resumed = SyntheticBatchSource((batch, batch, batch))
    resumed.load_state_dict(source.state_dict())
    assert resumed.next_batch() == batch
