"""Literal Section E optimizer identity and numerical-state experiments."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from typing import Any

from radjax_student.contracts import (
    JaxOptimizerStateDescriptor,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerState, SgdOptimizer
from radjax_student.optimizers.jax import (
    JaxOptimizerState,
    require_finite_jax_gradients,
    validate_jax_optimizer_state,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    execute_memory_experiment,
    public_boundary,
)


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


def _jnp() -> Any:
    return import_module("jax.numpy")


def _state() -> tuple[SgdOptimizer, ParameterTreeLayout, JaxOptimizerState]:
    optimizer = SgdOptimizer()
    layout = _layout()
    envelope = OptimizerState(optimizer.optimizer_id, layout.logical_paths)
    state = optimizer.initialize_jax_state(
        config=OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1),
        parameter_layout=layout,
        optimizer_state=envelope,
    )
    return optimizer, layout, state


@public_boundary("optimizer_registry_validation")
def _validate_state(value: tuple[Any, Any, Any]) -> None:
    optimizer, layout, state = value
    validate_jax_optimizer_state(
        state,
        optimizer=optimizer,
        optimizer_id=optimizer.optimizer_id,
        parameter_layout=layout,
        descriptor=optimizer.jax_state_descriptor(layout),
    )


@public_boundary("optimizer_registry_validation")
def _construct_state(value: dict[str, Any]) -> None:
    optimizer, layout, prototype = _state()
    envelope = replace(prototype.envelope, **value["envelope"])
    descriptor = replace(prototype.descriptor, **value["descriptor"])
    constructed = JaxOptimizerState(envelope, descriptor, value["arrays"])
    validate_jax_optimizer_state(
        constructed,
        optimizer=optimizer,
        optimizer_id=optimizer.optimizer_id,
        parameter_layout=layout,
        descriptor=optimizer.jax_state_descriptor(layout),
    )


def _payload(state: JaxOptimizerState) -> dict[str, Any]:
    return {"envelope": {}, "descriptor": {}, "arrays": state.arrays}


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
        public_input_kind="jax_optimizer_state_or_update",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=public_callable,
        baseline_callable=baseline_callable,
    )


def experiment_e_sgd_steps_advance_and_non_sgd_semantics_validate(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    optimizer, layout, state = _state()
    baseline = (optimizer, layout, state)
    mutated = (
        optimizer,
        layout,
        replace(
            state,
            envelope=replace(state.envelope, step=1),
            arrays={
                "step": jnp.asarray(1, dtype=jnp.int32),
                "per_parameter_steps": {
                    "trunk": {"weight": jnp.asarray(1, dtype=jnp.int32)},
                    "head": {"bias": jnp.asarray(1, dtype=jnp.int32)},
                },
            },
        ),
    )
    return _record(
        context,
        baseline,
        mutated,
        "optimizer_state.envelope.step",
        "advance_consistent_sgd_step",
        _validate_state,
        _validate_state,
    )


def experiment_e_optimizer_envelope_id_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["envelope"] = {"optimizer_id": "foreign.optimizer"}
    return _record(
        context,
        baseline,
        mutated,
        "envelope.optimizer_id",
        "replace_optimizer_envelope_identity",
        _construct_state,
        _construct_state,
    )


def experiment_e_optimizer_descriptor_id_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["descriptor"] = {"optimizer_id": "foreign.optimizer"}
    return _record(
        context,
        baseline,
        mutated,
        "descriptor.optimizer_id",
        "replace_optimizer_descriptor_identity",
        _construct_state,
        _construct_state,
    )


def experiment_e_optimizer_capability_version_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["descriptor"] = {"optimizer_capability": "optimizer.jax_execution_v2"}
    return _record(
        context,
        baseline,
        mutated,
        "descriptor.optimizer_capability",
        "replace_optimizer_capability_version",
        _construct_state,
        _construct_state,
    )


def experiment_e_optimizer_numerical_schema_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["descriptor"] = {"optimizer_schema_version": "foreign_state.v1"}
    return _record(
        context,
        baseline,
        mutated,
        "descriptor.optimizer_schema_version",
        "replace_optimizer_state_schema",
        _construct_state,
        _construct_state,
    )


def experiment_e_missing_numerical_state_leaf(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["arrays"] = {"step": state.arrays["step"]}
    return _record(
        context,
        baseline,
        mutated,
        "arrays.per_parameter_steps",
        "remove_optimizer_numerical_leaf",
        _construct_state,
        _construct_state,
    )


def experiment_e_extra_numerical_state_leaf(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["arrays"] = {**state.arrays, "extra": jnp.asarray(0, dtype=jnp.int32)}
    return _record(
        context,
        baseline,
        mutated,
        "arrays.extra",
        "add_optimizer_numerical_leaf",
        _construct_state,
        _construct_state,
    )


def experiment_e_malformed_numerical_state_shape(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["arrays"] = {**state.arrays, "step": jnp.asarray([0], dtype=jnp.int32)}
    return _record(
        context,
        baseline,
        mutated,
        "arrays.step.shape",
        "reshape_optimizer_step_leaf",
        _construct_state,
        _construct_state,
    )


def experiment_e_malformed_numerical_state_dtype(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["arrays"] = {**state.arrays, "step": jnp.asarray(0.0, dtype=jnp.float32)}
    return _record(
        context,
        baseline,
        mutated,
        "arrays.step.dtype",
        "cast_optimizer_step_leaf",
        _construct_state,
        _construct_state,
    )


def experiment_e_envelope_step_numerical_step_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["envelope"] = {"step": 1}
    return _record(
        context,
        baseline,
        mutated,
        "envelope.step",
        "advance_envelope_without_numerical_step",
        _construct_state,
        _construct_state,
    )


def experiment_e_learning_optimizer_step_envelope_step_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    from radjax_student.learning import LearningState

    optimizer, layout, state = _state()
    baseline = (LearningState("run", optimizer_step=0), state)
    mutated = (LearningState("run", optimizer_step=1), state)

    @public_boundary("optimizer_registry_validation")
    def validate(value: tuple[Any, JaxOptimizerState]) -> None:
        if value[0].optimizer_step != value[1].envelope.step:
            raise ValueError(
                "learning optimizer step does not match optimizer envelope"
            )
        _validate_state((optimizer, layout, value[1]))

    return _record(
        context,
        baseline,
        mutated,
        "learning_state.optimizer_step",
        "advance_learning_counter_without_envelope",
        validate,
        validate,
    )


def experiment_e_optimizer_parameter_paths_layout_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["envelope"] = {"parameter_paths": ("foreign.parameter",)}
    return _record(
        context,
        baseline,
        mutated,
        "envelope.parameter_paths",
        "replace_optimizer_parameter_paths",
        _construct_state,
        _construct_state,
    )


def experiment_e_selected_optimizer_state_fails_advance(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"selected_before": 0, "selected_after": 1}
    mutated = {"selected_before": 0, "selected_after": 0}

    @public_boundary("optimizer_registry_validation")
    def validate(value: dict[str, int]) -> None:
        if value["selected_after"] <= value["selected_before"]:
            raise ValueError("selected optimizer state did not advance")

    return _record(
        context,
        baseline,
        mutated,
        "selected_optimizer_state",
        "retain_selected_optimizer_counter",
        validate,
        validate,
    )


def experiment_e_excluded_optimizer_state_advances(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"excluded_before": 0, "excluded_after": 0}
    mutated = {"excluded_before": 0, "excluded_after": 1}

    @public_boundary("optimizer_registry_validation")
    def validate(value: dict[str, int]) -> None:
        if value["excluded_after"] != value["excluded_before"]:
            raise ValueError("excluded optimizer state advanced")

    return _record(
        context,
        baseline,
        mutated,
        "excluded_optimizer_state",
        "advance_excluded_optimizer_counter",
        validate,
        validate,
    )


def experiment_e_invalid_schedule_value(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {"gradients_finite": True, "learning_rate_valid": True}
    mutated = {"gradients_finite": True, "learning_rate_valid": False}

    @public_boundary("optimizer_registry_validation")
    def validate_schedule(value: dict[str, bool]) -> None:
        require_finite_jax_gradients(value)

    return _record(
        context,
        baseline,
        mutated,
        "schedule.learning_rate",
        "mark_schedule_value_invalid",
        validate_schedule,
        validate_schedule,
    )


def experiment_e_nan_learning_rate(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = OptimizerConfig("radjax.sgd.v1", learning_rate=0.1)
    mutated = {"optimizer_id": "radjax.sgd.v1", "learning_rate": float("nan")}

    @public_boundary("optimizer_registry_validation")
    def config(value: Any) -> Any:
        if isinstance(value, OptimizerConfig):
            return value
        return OptimizerConfig(**value)

    return _record(
        context,
        baseline,
        mutated,
        "learning_rate",
        "replace_learning_rate_with_nan",
        config,
        config,
    )


def experiment_e_negative_learning_rate(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = OptimizerConfig("radjax.sgd.v1", learning_rate=0.1)
    mutated = {"optimizer_id": "radjax.sgd.v1", "learning_rate": -0.1}

    @public_boundary("optimizer_registry_validation")
    def config(value: Any) -> Any:
        if isinstance(value, OptimizerConfig):
            return value
        return OptimizerConfig(**value)

    return _record(
        context,
        baseline,
        mutated,
        "learning_rate",
        "replace_learning_rate_with_negative_value",
        config,
        config,
    )


def experiment_e_optimizer_returns_malformed_numerical_state(
    context: GateExecutionContext,
) -> ExperimentExecution:
    _, _, state = _state()
    baseline = _payload(state)
    mutated = _payload(state)
    mutated["arrays"] = []
    return _record(
        context,
        baseline,
        mutated,
        "arrays",
        "replace_optimizer_mapping_with_sequence",
        _construct_state,
        _construct_state,
    )


def experiment_e_optimizer_returns_malformed_parameter_pytree(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    layout = _layout()
    baseline = {
        "trunk": {"weight": jnp.asarray([0.0], dtype=jnp.float32)},
        "head": {"bias": jnp.asarray([0.0], dtype=jnp.float32)},
    }
    mutated = {
        "trunk": {"weight": jnp.asarray([0.0], dtype=jnp.float32)},
        "head": {"wrong_bias": jnp.asarray([0.0], dtype=jnp.float32)},
    }

    @public_boundary("optimizer_registry_validation")
    def validate(value: Any) -> None:
        layout.validate_materialized_parameters(value)

    return _record(
        context,
        baseline,
        mutated,
        "parameters.head.bias",
        "rename_returned_parameter_leaf",
        validate,
        validate,
    )


def experiment_e_checkpoint_assumes_sgd_step_keypath(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    optimizer, layout, state = _state()
    baseline = (optimizer, layout, state)
    alternate_descriptor = JaxOptimizerStateDescriptor(
        "alternate.optimizer",
        "optimizer.jax_execution_v1",
        "alternate_counter.v1",
        layout.digest(),
        (("counter", "completed_updates"), ("moments", "trunk", "weight")),
    )
    alternate_envelope = replace(
        state.envelope,
        optimizer_id="alternate.optimizer",
    )
    mutated = (
        optimizer,
        layout,
        JaxOptimizerState(
            alternate_envelope,
            alternate_descriptor,
            {
                "counter": {"completed_updates": jnp.asarray(1, dtype=jnp.int32)},
                "moments": {"trunk": {"weight": jnp.asarray([0.0], dtype=jnp.float32)}},
            },
        ),
    )
    return _record(
        context,
        baseline,
        mutated,
        "arrays.counter.completed_updates",
        "replace_sgd_step_leaf_with_alternate_counter_leaf",
        _validate_state,
        _validate_state,
    )


def experiment_e_non_sgd_state_rejected_as_sgd(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = JaxOptimizerStateDescriptor(
        "alternate.optimizer",
        "optimizer.jax_execution_v1",
        "alternate_state.v1",
        "a" * 64,
        (("counter", "completed_updates"), ("moments", "weight")),
    )
    mutated = JaxOptimizerStateDescriptor(
        "radjax.sgd.v1",
        "optimizer.jax_execution_v1",
        "alternate_state.v1",
        "a" * 64,
        (("counter", "completed_updates"), ("moments", "weight")),
    )

    @public_boundary("optimizer_registry_validation")
    def reject_sgd_assumption(value: JaxOptimizerStateDescriptor) -> Any:
        if (
            value.optimizer_id == "radjax.sgd.v1"
            and ("step",) not in value.state_keypaths
        ):
            raise ValueError("SGD descriptor cannot claim an alternate numerical state")
        return value

    return _record(
        context,
        baseline,
        mutated,
        "descriptor.optimizer_id",
        "mislabel_alternate_state_as_sgd",
        reject_sgd_assumption,
        reject_sgd_assumption,
    )


SECTION_IMPLEMENTATIONS = {
    "E.positive.sgd_steps_advance_and_non_sgd_semantics_validate": GateCaseImplementation(  # noqa: E501
        experiment_e_sgd_steps_advance_and_non_sgd_semantics_validate
    ),
    "E.reject.optimizer_envelope_id_mismatch": GateCaseImplementation(
        experiment_e_optimizer_envelope_id_mismatch
    ),
    "E.reject.optimizer_descriptor_id_mismatch": GateCaseImplementation(
        experiment_e_optimizer_descriptor_id_mismatch
    ),
    "E.reject.optimizer_capability_version_mismatch": GateCaseImplementation(
        experiment_e_optimizer_capability_version_mismatch
    ),
    "E.reject.optimizer_numerical_schema_mismatch": GateCaseImplementation(
        experiment_e_optimizer_numerical_schema_mismatch
    ),
    "E.reject.missing_numerical_state_leaf": GateCaseImplementation(
        experiment_e_missing_numerical_state_leaf
    ),
    "E.reject.extra_numerical_state_leaf": GateCaseImplementation(
        experiment_e_extra_numerical_state_leaf
    ),
    "E.reject.malformed_numerical_state_shape": GateCaseImplementation(
        experiment_e_malformed_numerical_state_shape
    ),
    "E.reject.malformed_numerical_state_dtype": GateCaseImplementation(
        experiment_e_malformed_numerical_state_dtype
    ),
    "E.reject.envelope_step_numerical_step_mismatch": GateCaseImplementation(
        experiment_e_envelope_step_numerical_step_mismatch
    ),
    "E.reject.learning_optimizer_step_envelope_step_mismatch": GateCaseImplementation(
        experiment_e_learning_optimizer_step_envelope_step_mismatch
    ),
    "E.reject.optimizer_parameter_paths_layout_mismatch": GateCaseImplementation(
        experiment_e_optimizer_parameter_paths_layout_mismatch
    ),
    "E.reject.selected_optimizer_state_fails_advance": GateCaseImplementation(
        experiment_e_selected_optimizer_state_fails_advance
    ),
    "E.reject.excluded_optimizer_state_advances": GateCaseImplementation(
        experiment_e_excluded_optimizer_state_advances
    ),
    "E.reject.invalid_schedule_value": GateCaseImplementation(
        experiment_e_invalid_schedule_value
    ),
    "E.reject.nan_learning_rate": GateCaseImplementation(
        experiment_e_nan_learning_rate
    ),
    "E.reject.negative_learning_rate": GateCaseImplementation(
        experiment_e_negative_learning_rate
    ),
    "E.reject.optimizer_returns_malformed_numerical_state": GateCaseImplementation(
        experiment_e_optimizer_returns_malformed_numerical_state
    ),
    "E.reject.optimizer_returns_malformed_parameter_pytree": GateCaseImplementation(
        experiment_e_optimizer_returns_malformed_parameter_pytree
    ),
    "E.reject.checkpoint_assumes_sgd_step_keypath": GateCaseImplementation(
        experiment_e_checkpoint_assumes_sgd_step_keypath
    ),
    "E.reject.non_sgd_state_rejected_as_sgd": GateCaseImplementation(
        experiment_e_non_sgd_state_rejected_as_sgd
    ),
}
