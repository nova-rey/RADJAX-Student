"""Literal Section C batch and objective binding experiments.

Each experiment changes the concrete batch, resolved surface, or objective input
named by its inventory record.  The helpers below are deliberately unaware of
the inventory and only expose existing public contract boundaries.
"""

from __future__ import annotations

from typing import Any

from radjax_student.architecture import ArchitectureConfig, ForwardResult
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.contracts import LearningBatch, MetricRecord, ObjectiveScope
from radjax_student.learning.jax_batch import FiniteJsonJaxBatchMaterializer
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_memory_experiment,
    public_boundary,
)


def _batch(batch_id: str, *, x: object = 1.0, y: object = 2.0) -> LearningBatch:
    return LearningBatch(batch_id, inputs={"x": x}, targets={"y": y})


@public_boundary("learning_batch_validation")
def _construct_batch(value: dict[str, Any]) -> LearningBatch:
    return LearningBatch.from_dict(value)


@public_boundary("learning_batch_validation")
def _validate_token_batch(value: LearningBatch) -> Any:
    plugin = FakeArchitecturePlugin()
    result = plugin.validate_batch(value, ArchitectureConfig(plugin.architecture_id))
    if result.status != "pass":
        raise ValueError("architecture batch validation rejected the concrete batch")
    return result


@public_boundary("learning_batch_validation")
def _materialize_batch(value: LearningBatch) -> Any:
    return FiniteJsonJaxBatchMaterializer().materialize(value)


@public_boundary("learning_batch_validation")
def _surface(value: tuple[ForwardResult, str]) -> Any:
    return value[0].surface_for(value[1])


@public_boundary("learning_batch_validation")
def _metric(value: tuple[str, float]) -> MetricRecord:
    return MetricRecord(value[0], value[1], 0)


def _record(
    context: GateExecutionContext,
    baseline: Any,
    mutated: Any,
    path: str,
    operation: str,
    public_callable: Any,
    baseline_callable: Any | None = None,
) -> ExperimentExecution:
    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="learning_batch_or_objective_input",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=public_callable,
        baseline_callable=baseline_callable,
    )


def experiment_c_validated_learning_batch_is_materialized_and_executed(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _batch("batch-one")
    mutated = _batch("batch-two")
    return _record(
        context,
        baseline,
        mutated,
        "batch_id",
        "replace_valid_batch_identity",
        _materialize_batch,
        _materialize_batch,
    )


def experiment_c_malformed_learning_batch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"batch_id": "valid", "inputs": {"x": 1.0}, "targets": {"y": 2.0}}
    mutated = {"batch_id": "", "inputs": {"x": 1.0}, "targets": {"y": 2.0}}
    return _record(
        context,
        baseline,
        mutated,
        "batch_id",
        "empty_batch_id",
        _construct_batch,
        _construct_batch,
    )


def experiment_c_nonfinite_batch_value(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"batch_id": "finite", "inputs": {"x": 1.0}, "targets": {"y": 2.0}}
    mutated = {
        "batch_id": "finite",
        "inputs": {"x": float("nan")},
        "targets": {"y": 2.0},
    }
    return _record(
        context,
        baseline,
        mutated,
        "inputs.x",
        "replace_finite_batch_value_with_nan",
        _construct_batch,
        _construct_batch,
    )


def experiment_c_missing_required_batch_input(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = LearningBatch(
        "token-batch",
        inputs={"token_ids": {"rank": 2, "sequence_length": 1}},
        targets={"y": 1},
    )
    mutated = LearningBatch("token-batch", inputs={}, targets={"y": 1})
    return _record(
        context,
        baseline,
        mutated,
        "inputs.token_ids",
        "remove_required_token_ids_input",
        _validate_token_batch,
        _validate_token_batch,
    )


def experiment_c_missing_required_target(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _batch("target-batch")
    mutated = LearningBatch("target-batch", inputs={"x": 1.0}, targets={})

    @public_boundary("learning_batch_validation")
    def require_target(value: LearningBatch) -> Any:
        return value.targets["y"]

    return _record(
        context,
        baseline,
        mutated,
        "targets.y",
        "remove_required_target",
        require_target,
        require_target,
    )


def experiment_c_validated_batch_a_executed_batch_b(
    context: GateExecutionContext,
) -> ExperimentExecution:
    from radjax_student.learning.jax_core import JaxBatch
    from radjax_student.steps.jax_step import validate_jax_batch_binding

    batch_a = _batch("batch-a", x=[1.0], y=[2.0])
    batch_b = _batch("batch-b", x=[3.0], y=[4.0])
    baseline = (batch_a, FiniteJsonJaxBatchMaterializer().materialize(batch_a))
    executed_b = FiniteJsonJaxBatchMaterializer().materialize(batch_b)
    mutated = (
        batch_a,
        JaxBatch(
            executed_b.inputs,
            executed_b.targets,
            executed_b.weights,
            executed_b.source_batch_digest,
        ),
    )

    @public_boundary("learning_batch_validation")
    def bind(value: tuple[LearningBatch, JaxBatch]) -> None:
        validate_jax_batch_binding(value[0], value[1])

    return _record(
        context,
        baseline,
        mutated,
        "jax_batch.source_batch_digest",
        "pair_validated_batch_a_with_materialized_batch_b",
        bind,
        bind,
    )


def experiment_c_materializer_foreign_source_digest(
    context: GateExecutionContext,
) -> ExperimentExecution:
    from radjax_student.learning.jax_core import JaxBatch
    from radjax_student.steps.jax_step import validate_jax_batch_binding

    learning_batch = _batch("source-digest", x=[1.0], y=[2.0])
    correct = FiniteJsonJaxBatchMaterializer().materialize(learning_batch)
    baseline = (learning_batch, correct)
    mutated_batch = JaxBatch(correct.inputs, correct.targets, correct.weights, "f" * 64)
    mutated = (learning_batch, mutated_batch)

    @public_boundary("learning_batch_validation")
    def bind(value: tuple[LearningBatch, JaxBatch]) -> None:
        validate_jax_batch_binding(value[0], value[1])

    return _record(
        context,
        baseline,
        mutated,
        "jax_batch.source_batch_digest",
        "replace_materializer_source_digest",
        bind,
        bind,
    )


def experiment_c_objective_id_drift(
    context: GateExecutionContext,
) -> ExperimentExecution:
    from radjax_student.learning.jax_core import JaxObjectiveConfig

    baseline = {"objective_id": "linear.mse.v1"}
    mutated = {"objective_id": ""}

    @public_boundary("learning_batch_validation")
    def construct(value: dict[str, str]) -> Any:
        return JaxObjectiveConfig(value["objective_id"])

    return _record(
        context,
        baseline,
        mutated,
        "objective_id",
        "clear_objective_identity",
        construct,
        construct,
    )


def experiment_c_objective_surface_drift(
    context: GateExecutionContext,
) -> ExperimentExecution:
    forward = ForwardResult(outputs=1.0, surface_values={"trunk_output": 2.0})
    baseline = (forward, "final_output")
    mutated = (forward, "missing_surface")
    return _record(
        context,
        baseline,
        mutated,
        "resolved_surface_id",
        "replace_with_missing_surface",
        _surface,
        _surface,
    )


def experiment_c_objective_scope_drift(
    context: GateExecutionContext,
) -> ExperimentExecution:
    forward = ForwardResult(outputs=1.0)
    baseline = (forward, ObjectiveScope())
    mutated = (forward, ObjectiveScope("intermediate_surface", "unknown"))

    @public_boundary("learning_batch_validation")
    def legacy_surface(value: tuple[ForwardResult, ObjectiveScope]) -> Any:
        return value[0].surface_for_legacy_scope(value[1])

    return _record(
        context,
        baseline,
        mutated,
        "objective_scope.target_id",
        "replace_with_unavailable_objective_scope",
        legacy_surface,
        legacy_surface,
    )


def experiment_c_target_shape_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"prediction": [1.0, 2.0], "target": [1.0, 2.0]}
    mutated = {"prediction": [1.0, 2.0], "target": [1.0]}

    @public_boundary("learning_batch_validation")
    def shape_check(value: dict[str, list[float]]) -> Any:
        if len(value["prediction"]) != len(value["target"]):
            raise ValueError("objective target shape does not match selected surface")
        return value

    return _record(
        context,
        baseline,
        mutated,
        "targets.shape",
        "remove_one_target_element",
        shape_check,
        shape_check,
    )


def experiment_c_nonfinite_objective_metric(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = ("mse", 1.0)
    mutated = ("mse", float("inf"))
    return _record(
        context,
        baseline,
        mutated,
        "objective_metrics.mse",
        "replace_finite_metric_with_infinity",
        _metric,
        _metric,
    )


def experiment_c_undeclared_architecture_surface(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = ForwardResult(outputs=1.0, surface_values={"declared": 2.0})
    mutated = ForwardResult(outputs=1.0, surface_values={"undeclared": 2.0})

    @public_boundary("learning_batch_validation")
    def declared_surface(value: ForwardResult) -> Any:
        return value.surface("declared")

    return _record(
        context,
        baseline,
        mutated,
        "surface_values.declared",
        "replace_declared_surface_with_undeclared_surface",
        declared_surface,
        declared_surface,
    )


def experiment_c_objective_consumes_parameters(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"surface": {"value": 1.0}}
    mutated = {"surface": {"parameters": {"trunk": 1.0}}}

    class SurfaceOnlyObjective:
        def evaluate(self, surface: dict[str, Any]) -> Any:
            return surface["value"]

    objective = SurfaceOnlyObjective()

    @public_boundary("learning_batch_validation")
    def evaluate(value: dict[str, Any]) -> Any:
        return objective.evaluate(value["surface"])

    return _record(
        context,
        baseline,
        mutated,
        "surface.parameters",
        "replace_objective_surface_with_raw_parameter_tree",
        evaluate,
        evaluate,
    )


SECTION_IMPLEMENTATIONS = {
    "C.positive.validated_learning_batch_is_materialized_and_executed": GateCaseImplementation(  # noqa: E501
        experiment_c_validated_learning_batch_is_materialized_and_executed
    ),
    "C.reject.malformed_learning_batch": GateCaseImplementation(
        experiment_c_malformed_learning_batch
    ),
    "C.reject.nonfinite_batch_value": GateCaseImplementation(
        experiment_c_nonfinite_batch_value
    ),
    "C.reject.missing_required_batch_input": GateCaseImplementation(
        experiment_c_missing_required_batch_input
    ),
    "C.reject.missing_required_target": GateCaseImplementation(
        experiment_c_missing_required_target
    ),
    "C.reject.validated_batch_a_executed_batch_b": GateCaseImplementation(
        experiment_c_validated_batch_a_executed_batch_b
    ),
    "C.reject.materializer_foreign_source_digest": GateCaseImplementation(
        experiment_c_materializer_foreign_source_digest
    ),
    "C.reject.objective_id_drift": GateCaseImplementation(
        experiment_c_objective_id_drift
    ),
    "C.reject.objective_surface_drift": GateCaseImplementation(
        experiment_c_objective_surface_drift
    ),
    "C.reject.objective_scope_drift": GateCaseImplementation(
        experiment_c_objective_scope_drift
    ),
    "C.reject.target_shape_mismatch": GateCaseImplementation(
        experiment_c_target_shape_mismatch
    ),
    "C.reject.nonfinite_objective_metric": GateCaseImplementation(
        experiment_c_nonfinite_objective_metric
    ),
    "C.reject.undeclared_architecture_surface": GateCaseImplementation(
        experiment_c_undeclared_architecture_surface
    ),
    "C.reject.objective_consumes_parameters": GateCaseImplementation(
        experiment_c_objective_consumes_parameters
    ),
}
