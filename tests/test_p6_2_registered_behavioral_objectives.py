"""P6.2 registry plugins preserve P5.5 B=1 behavioral objective semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.architecture import ArchitectureRegistry  # noqa: E402
from radjax_student.architecture.rwkv7_reference import (  # noqa: E402
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    configurable_architecture_config,
    register_rwkv7_reference,
)
from radjax_student.behavior import (  # noqa: E402
    BehaviorJaxBatchMaterializerV1,
    corridor_objective_v1,
    corridor_source_unit_learning_batch_v1,
    exemplar_coarse_cross_entropy_v1,
    exemplar_source_unit_learning_batch_v1,
    materialize_behavioral_batches_v1,
)
from radjax_student.contracts import (  # noqa: E402
    ObjectiveConfig,
    ObjectiveScope,
    UpdateScope,
)
from radjax_student.learning import (  # noqa: E402
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningState,
    assemble_jax_learning_lifecycle,
)
from radjax_student.objectives import build_default_objective_registry  # noqa: E402
from radjax_student.objectives.behavioral import (  # noqa: E402
    BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY,
    BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY,
    BEHAVIORAL_OBJECTIVE_CONFIG,
)
from radjax_student.optimizers import (  # noqa: E402
    OptimizerConfig,
    OptimizerRegistry,
    SgdOptimizer,
)
from radjax_student.runtime import (  # noqa: E402
    RuntimeConfig,
    build_default_runtime_registry,
)
from tests.test_p5_4_behavior_materialization import _projection  # noqa: E402

pytestmark = pytest.mark.jax


def test_p6_2_registered_corridor_objective_matches_one_p5_5_coordinate():
    batches = materialize_behavioral_batches_v1(_projection())
    original = batches.training_corridor
    coordinate, index, row = _corridor_coordinate(original)
    unit = corridor_source_unit_learning_batch_v1(
        batches, partition="training", coordinate=coordinate
    )
    materialized = BehaviorJaxBatchMaterializerV1().materialize(unit)
    reduced = replace(
        original,
        example_ids=(coordinate[0],),
        input_ids=original.input_ids[row : row + 1],
        attention_mask=original.attention_mask[row : row + 1],
        example_indices=original.example_indices[index : index + 1],
        positions=original.positions[index : index + 1],
        mode_ids=original.mode_ids[index : index + 1],
        assignment_weights=original.assignment_weights[index : index + 1],
    )
    logits = jnp.asarray(
        np.arange(16, dtype=np.float32).reshape(1, 4, 4) / np.float32(10)
    )
    expected_loss, expected_metrics = corridor_objective_v1(logits, reduced)
    plugin = (
        build_default_objective_registry()
        .select(BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY)
        .plugin
    )
    loss, metrics = plugin.evaluate_jax(
        surface=logits,
        targets=materialized.targets,
        weights=materialized.weights,
        config=ObjectiveConfig(
            BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY, BEHAVIORAL_OBJECTIVE_CONFIG
        ),
    )
    assert float(loss) == pytest.approx(float(expected_loss))
    assert dict(metrics).keys() == dict(expected_metrics).keys()
    for name in metrics:
        assert float(metrics[name]) == pytest.approx(float(expected_metrics[name]))


def test_p6_2_registered_exemplar_objective_matches_one_p5_5_passport():
    batches = materialize_behavioral_batches_v1(_projection())
    original = batches.training_exemplars
    row, passport = next(iter(enumerate(original.passports)))
    key = (
        str(passport["selected_example_id"]),
        int(passport["selected_position"]),
        str(passport["corridor_fingerprint_id"]),
    )
    unit = exemplar_source_unit_learning_batch_v1(
        batches, partition="training", passport_key=key
    )
    materialized = BehaviorJaxBatchMaterializerV1().materialize(unit)
    reduced = replace(
        original,
        example_ids=original.example_ids[row : row + 1],
        input_ids=original.input_ids[row : row + 1],
        attention_mask=original.attention_mask[row : row + 1],
        selected_example_indices=original.selected_example_indices[row : row + 1],
        selected_positions=original.selected_positions[row : row + 1],
        sparse_targets=original.sparse_targets[row : row + 1],
        passports=original.passports[row : row + 1],
    )
    logits = jnp.asarray(
        np.arange(16, dtype=np.float32).reshape(1, 4, 4) / np.float32(10)
    )
    expected_loss, expected_metrics = exemplar_coarse_cross_entropy_v1(logits, reduced)
    plugin = (
        build_default_objective_registry()
        .select(BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY)
        .plugin
    )
    loss, metrics = plugin.evaluate_jax(
        surface=logits,
        targets=materialized.targets,
        weights=materialized.weights,
        config=ObjectiveConfig(
            BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY, BEHAVIORAL_OBJECTIVE_CONFIG
        ),
    )
    assert float(loss) == pytest.approx(float(expected_loss))
    assert dict(metrics).keys() == dict(expected_metrics).keys()
    assert float(metrics["exemplar.coarse_cross_entropy"]) == pytest.approx(
        float(expected_metrics["exemplar.coarse_cross_entropy"])
    )


def test_p6_2_behavioral_source_unit_reaches_generic_rwkv_lifecycle() -> None:
    """The registered behavior seam, rather than a test batch, drives RWKV."""

    projection = _projection()
    batches = materialize_behavioral_batches_v1(projection)
    coordinate, _, _ = _corridor_coordinate(batches.training_corridor)
    batch = corridor_source_unit_learning_batch_v1(
        batches, partition="training", coordinate=coordinate
    )
    language = projection.language
    config = configurable_architecture_config(
        language.vocabulary.vocabulary_size,
        4,
        tokenizer=language.tokenizer,
        vocabulary=language.vocabulary,
        special_tokens=language.special_tokens,
    )
    architectures = ArchitectureRegistry()
    register_rwkv7_reference(architectures)
    optimizers = OptimizerRegistry()
    optimizers.register(SgdOptimizer())
    assembled = assemble_jax_learning_lifecycle(
        JaxLearningAssemblyRequest(
            architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
            architecture_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
            architecture_config=config,
            objective_identity=BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY,
            objective_config=ObjectiveConfig(
                BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY, BEHAVIORAL_OBJECTIVE_CONFIG
            ),
            optimizer_id="sgd.v1",
            optimizer_version=1,
            optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.01),
            runtime_backend_id="jax",
            runtime_implementation_version="p2.9",
            runtime_config=RuntimeConfig(
                backend_id="jax",
                platform_preference="cpu",
                precision_policy="float32",
                placement_policy="single_device",
                compilation_policy="eager",
                distributed_policy="disabled",
                fallback_policy="disallowed",
                seed=62,
            ),
            root_seed=62,
            learning_state=LearningState(
                "p6-2-behavioral-lifecycle",
                active_update_scope=UpdateScope(),
                active_objective_scope=ObjectiveScope(),
            ),
            batch_materializer_id="behavior_jax_batch_materializer.v1",
        ),
        registries=JaxLearningAssemblyRegistries(
            architectures,
            build_default_objective_registry(),
            optimizers,
            build_default_runtime_registry(),
            {"behavior_jax_batch_materializer.v1": BehaviorJaxBatchMaterializerV1()},
        ),
    )
    lifecycle = assembled.loop_executor.lifecycle
    execution = assembled.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        objective=lifecycle.objective_selection,
        batch=batch,
    )

    assert execution.runtime_result.status == "pass"
    assert execution.result.loss is not None
    assert execution.result.changed_parameter_paths
    assert (
        execution.runtime_result.callable_reference.callable_id
        == "radjax.learning.generic_jax_step"
    )


def _corridor_coordinate(batch):
    global_rows = tuple(sorted({int(value) for value in batch.example_indices}))
    ids = dict(zip(global_rows, batch.example_ids, strict=True))
    index = 0
    global_row = int(batch.example_indices[index])
    return (
        (ids[global_row], int(batch.positions[index]), int(batch.mode_ids[index])),
        index,
        batch.example_ids.index(ids[global_row]),
    )
