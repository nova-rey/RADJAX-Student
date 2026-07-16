"""Literal Section B parameter-layout and architecture-scope experiments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.contracts import (
    ObjectiveScope,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
    UpdateScope,
)
from radjax_student.learning import LearningStepResult
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
                "head.bias",
                ("head", "bias"),
                (1,),
                "float32",
                "output_head",
                ("head",),
            ),
        ),
    )


def _jnp() -> Any:
    return import_module("jax.numpy")


@public_boundary("parameter_layout_validation")
def _validate_materialized(payload: tuple[ParameterTreeLayout, Any]) -> None:
    layout, parameters = payload
    layout.validate_materialized_parameters(parameters)


@public_boundary("parameter_layout_validation")
def _construct_layout(
    payload: tuple[ParameterTreeLayoutEntry, ...],
) -> ParameterTreeLayout:
    return ParameterTreeLayout("test.architecture.v1", payload)


@public_boundary("parameter_layout_validation")
def _construct_entry(payload: dict[str, Any]) -> ParameterTreeLayoutEntry:
    return ParameterTreeLayoutEntry(**payload)


@public_boundary("parameter_layout_validation")
def _resolve_update(payload: UpdateScope) -> Any:
    plugin = FakeArchitecturePlugin()
    return plugin.resolve_update_scope(payload, plugin.describe_parameters())


@public_boundary("parameter_layout_validation")
def _resolve_objective(payload: ObjectiveScope) -> Any:
    plugin = FakeArchitecturePlugin()
    return plugin.resolve_objective_scope(payload, plugin.architecture_metadata())


@public_boundary("parameter_layout_validation")
def _construct_step_result(payload: dict[str, Any]) -> LearningStepResult:
    return LearningStepResult(**payload)


@public_boundary("parameter_layout_validation")
def _validate_update(payload: dict[str, Any]) -> None:
    # Keep passive final-gate imports JAX-free. The public boundary is loaded
    # only when this JAX execution experiment actually runs.
    from radjax_student.steps.jax_step import validate_jax_update_evidence

    validate_jax_update_evidence(**payload)


def _record(
    context: GateExecutionContext,
    baseline: Any,
    mutated: Any,
    path: str,
    operation: str,
    callable_: Any,
) -> ExperimentExecution:
    return execute_memory_experiment(
        context,
        baseline=baseline,
        mutated=mutated,
        public_input_kind="parameter_layout_or_scope",
        canonical_path=path,
        operation=operation,
        value_summary={"path": path, "operation": operation},
        public_callable=callable_,
        baseline_callable=callable_,
    )


def _tree(trunk: Any, head: Any) -> dict[str, dict[str, Any]]:
    return {"trunk": {"weight": trunk}, "head": {"bias": head}}


def _valid_step_paths() -> dict[str, Any]:
    return {
        "status": "pass",
        "global_step_before": 0,
        "global_step_after": 1,
        "changed_parameter_paths": ("trunk.weight",),
        "unchanged_parameter_paths": ("head.bias",),
    }


def _valid_update_payload(before: Any, after: Any) -> dict[str, Any]:
    return {
        "before_parameters": before,
        "after_parameters": after,
        "parameter_layout": _layout(),
        "selected_paths": ("trunk.weight",),
        "changed_paths": ("trunk.weight",),
        "unchanged_paths": ("head.bias",),
        "optimizer_before": {
            "trunk.weight": before["trunk"]["weight"],
            "head.bias": before["head"]["bias"],
        },
        "optimizer_after": {
            "trunk.weight": after["trunk"]["weight"],
            "head.bias": after["head"]["bias"],
        },
    }


def experiment_b_trunk_changes_head_bias_unchanged_and_changed_mask_matches(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    before = _tree(
        jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    baseline_after = _tree(
        jnp.asarray([1.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    mutated_after = _tree(
        jnp.asarray([2.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    baseline = _valid_update_payload(before, baseline_after)
    mutated = _valid_update_payload(before, mutated_after)
    return _record(
        context,
        baseline,
        mutated,
        "update.trunk.weight",
        "validate_scoped_jax_update",
        _validate_update,
    )


def experiment_b_missing_parameter_leaf(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    layout = _layout()
    baseline = (
        layout,
        _tree(
            jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
        ),
    )
    mutated = (layout, {"trunk": {"weight": jnp.asarray([0.0], dtype=jnp.float32)}})
    return _record(
        context,
        baseline,
        mutated,
        "parameters.head.bias",
        "remove_parameter_leaf",
        _validate_materialized,
    )


def experiment_b_extra_parameter_leaf(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    layout = _layout()
    baseline = (
        layout,
        _tree(
            jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
        ),
    )
    mutated = (
        layout,
        {
            "trunk": {"weight": jnp.asarray([0.0], dtype=jnp.float32)},
            "head": {"bias": jnp.asarray([0.0], dtype=jnp.float32)},
            "extra": {"weight": jnp.asarray([0.0], dtype=jnp.float32)},
        },
    )
    return _record(
        context,
        baseline,
        mutated,
        "parameters.extra.weight",
        "add_parameter_leaf",
        _validate_materialized,
    )


def experiment_b_wrong_parameter_keypath(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    layout = _layout()
    baseline = (
        layout,
        _tree(
            jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
        ),
    )
    mutated = (
        layout,
        {
            "trunk": {"kernel": jnp.asarray([0.0], dtype=jnp.float32)},
            "head": {"bias": jnp.asarray([0.0], dtype=jnp.float32)},
        },
    )
    return _record(
        context,
        baseline,
        mutated,
        "parameters.trunk.kernel",
        "rename_parameter_keypath",
        _validate_materialized,
    )


def experiment_b_wrong_parameter_shape(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    layout = _layout()
    baseline = (
        layout,
        _tree(
            jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
        ),
    )
    mutated = (
        layout,
        _tree(
            jnp.asarray([[0.0]], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
        ),
    )
    return _record(
        context,
        baseline,
        mutated,
        "parameters.trunk.weight.shape",
        "reshape_parameter_leaf",
        _validate_materialized,
    )


def experiment_b_wrong_parameter_dtype(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    layout = _layout()
    baseline = (
        layout,
        _tree(
            jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
        ),
    )
    mutated = (
        layout,
        _tree(jnp.asarray([0], dtype=jnp.int32), jnp.asarray([0.0], dtype=jnp.float32)),
    )
    return _record(
        context,
        baseline,
        mutated,
        "parameters.trunk.weight.dtype",
        "cast_parameter_leaf",
        _validate_materialized,
    )


def experiment_b_catalog_layout_logical_path_mismatch(
    context: GateExecutionContext,
) -> ExperimentExecution:
    plugin = FakeArchitecturePlugin()
    baseline = (plugin, plugin.describe_parameters())
    catalog = plugin.describe_parameters()
    mutated = (plugin, type(catalog)("foreign.architecture.v1", catalog.parameters))

    @public_boundary("parameter_layout_validation")
    def validate_catalog(value: tuple[Any, Any]) -> Any:
        architecture, parameter_catalog = value
        return architecture.resolve_update_scope(UpdateScope(), parameter_catalog)

    return _record(
        context,
        baseline,
        mutated,
        "catalog.architecture_id",
        "replace_catalog_layout_identity",
        validate_catalog,
    )


def experiment_b_duplicate_logical_path(
    context: GateExecutionContext,
) -> ExperimentExecution:
    entries = _layout().entries
    baseline = entries
    mutated = (entries[0], entries[0])
    return _record(
        context,
        baseline,
        mutated,
        "entries.logical_path",
        "duplicate_logical_path",
        _construct_layout,
    )


def experiment_b_duplicate_physical_keypath(
    context: GateExecutionContext,
) -> ExperimentExecution:
    entries = _layout().entries
    duplicate = ParameterTreeLayoutEntry(
        "head.kernel", ("head", "bias"), (1,), "float32", "output_head"
    )
    baseline = entries
    mutated = (entries[0], duplicate)
    return _record(
        context,
        baseline,
        mutated,
        "entries.jax_keypath",
        "duplicate_physical_keypath",
        _construct_layout,
    )


def experiment_b_malformed_nested_layout_metadata(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {
        "logical_path": "trunk.weight",
        "jax_keypath": ("trunk", "weight"),
        "shape": (1,),
        "dtype": "float32",
        "role": "recurrent_block",
        "metadata": {"quantization": {"scheme": "none"}},
    }
    mutated = {
        "logical_path": "trunk.weight",
        "jax_keypath": ("trunk", "weight"),
        "shape": (1,),
        "dtype": "float32",
        "role": "recurrent_block",
        "metadata": {"quantization": {"scheme": object()}},
    }
    return _record(
        context,
        baseline,
        mutated,
        "metadata.quantization.scheme",
        "insert_non_json_nested_metadata",
        _construct_entry,
    )


def experiment_b_noncanonical_metadata_key(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = {
        "logical_path": "trunk.weight",
        "jax_keypath": ("trunk", "weight"),
        "shape": (1,),
        "dtype": "float32",
        "role": "recurrent_block",
        "metadata": {"canonical": True},
    }
    mutated = {
        "logical_path": "trunk.weight",
        "jax_keypath": ("trunk", "weight"),
        "shape": (1,),
        "dtype": "float32",
        "role": "recurrent_block",
        "metadata": {"": True},
    }
    return _record(
        context,
        baseline,
        mutated,
        "metadata.<empty>",
        "insert_noncanonical_metadata_key",
        _construct_entry,
    )


def experiment_b_unknown_update_region(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = UpdateScope("named_region", "trunk")
    mutated = UpdateScope("named_region", "unknown-region")
    return _record(
        context,
        baseline,
        mutated,
        "update_scope.region_id",
        "replace_with_unknown_region",
        _resolve_update,
    )


def experiment_b_unavailable_objective_surface(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = ObjectiveScope("intermediate_surface", "trunk_output")
    mutated = ObjectiveScope("intermediate_surface", "missing-output")
    return _record(
        context,
        baseline,
        mutated,
        "objective_scope.target_id",
        "replace_with_unavailable_surface",
        _resolve_objective,
    )


def experiment_b_explicit_unknown_parameter_path(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = UpdateScope("parameter_paths", parameter_paths=("trunk.weight",))
    mutated = UpdateScope("parameter_paths", parameter_paths=("missing.weight",))
    return _record(
        context,
        baseline,
        mutated,
        "update_scope.parameter_paths",
        "replace_with_unknown_parameter_path",
        _resolve_update,
    )


def experiment_b_overlapping_changed_unchanged_evidence(
    context: GateExecutionContext,
) -> ExperimentExecution:
    baseline = _valid_step_paths()
    mutated = _valid_step_paths()
    mutated["unchanged_parameter_paths"] = ("trunk.weight", "head.bias")
    return _record(
        context,
        baseline,
        mutated,
        "step.unchanged_parameter_paths",
        "overlap_changed_and_unchanged_paths",
        _construct_step_result,
    )


def experiment_b_excluded_parameter_changes(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    before = _tree(
        jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    after = _tree(
        jnp.asarray([1.0], dtype=jnp.float32), jnp.asarray([1.0], dtype=jnp.float32)
    )
    baseline = _valid_update_payload(
        before,
        _tree(
            jnp.asarray([1.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
        ),
    )
    mutated = _valid_update_payload(before, after)
    return _record(
        context,
        baseline,
        mutated,
        "parameters.head.bias",
        "change_excluded_parameter",
        _validate_update,
    )


def experiment_b_excluded_optimizer_state_changes(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    before = _tree(
        jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    after = _tree(
        jnp.asarray([1.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    baseline = _valid_update_payload(before, after)
    mutated = _valid_update_payload(before, after)
    mutated["optimizer_after"] = {
        "trunk.weight": jnp.asarray([1.0], dtype=jnp.float32),
        "head.bias": jnp.asarray([1.0], dtype=jnp.float32),
    }
    return _record(
        context,
        baseline,
        mutated,
        "optimizer_state.head.bias",
        "advance_excluded_optimizer_state",
        _validate_update,
    )


def experiment_b_selected_zero_change_claimed_changed(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    before = _tree(
        jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    baseline = _valid_update_payload(
        before,
        _tree(
            jnp.asarray([1.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
        ),
    )
    mutated = _valid_update_payload(before, before)
    return _record(
        context,
        baseline,
        mutated,
        "changed_paths.trunk.weight",
        "claim_zero_change_as_changed",
        _validate_update,
    )


def experiment_b_unselected_parameter_claimed_changed(
    context: GateExecutionContext,
) -> ExperimentExecution:
    jnp = _jnp()

    before = _tree(
        jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    valid_after = _tree(
        jnp.asarray([1.0], dtype=jnp.float32), jnp.asarray([0.0], dtype=jnp.float32)
    )
    baseline = _valid_update_payload(before, valid_after)
    mutated_after = _tree(
        jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([1.0], dtype=jnp.float32)
    )
    mutated = _valid_update_payload(before, mutated_after)
    mutated["changed_paths"] = ("head.bias",)
    mutated["unchanged_paths"] = ("trunk.weight",)
    return _record(
        context,
        baseline,
        mutated,
        "changed_paths.head.bias",
        "claim_unselected_parameter_as_changed",
        _validate_update,
    )


SECTION_IMPLEMENTATIONS = {
    "B.positive.trunk_changes_head_bias_unchanged_and_changed_mask_matches": GateCaseImplementation(  # noqa: E501
        experiment_b_trunk_changes_head_bias_unchanged_and_changed_mask_matches
    ),
    "B.reject.missing_parameter_leaf": GateCaseImplementation(
        experiment_b_missing_parameter_leaf
    ),
    "B.reject.extra_parameter_leaf": GateCaseImplementation(
        experiment_b_extra_parameter_leaf
    ),
    "B.reject.wrong_parameter_keypath": GateCaseImplementation(
        experiment_b_wrong_parameter_keypath
    ),
    "B.reject.wrong_parameter_shape": GateCaseImplementation(
        experiment_b_wrong_parameter_shape
    ),
    "B.reject.wrong_parameter_dtype": GateCaseImplementation(
        experiment_b_wrong_parameter_dtype
    ),
    "B.reject.catalog_layout_logical_path_mismatch": GateCaseImplementation(
        experiment_b_catalog_layout_logical_path_mismatch
    ),
    "B.reject.duplicate_logical_path": GateCaseImplementation(
        experiment_b_duplicate_logical_path
    ),
    "B.reject.duplicate_physical_keypath": GateCaseImplementation(
        experiment_b_duplicate_physical_keypath
    ),
    "B.reject.malformed_nested_layout_metadata": GateCaseImplementation(
        experiment_b_malformed_nested_layout_metadata
    ),
    "B.reject.noncanonical_metadata_key": GateCaseImplementation(
        experiment_b_noncanonical_metadata_key
    ),
    "B.reject.unknown_update_region": GateCaseImplementation(
        experiment_b_unknown_update_region
    ),
    "B.reject.unavailable_objective_surface": GateCaseImplementation(
        experiment_b_unavailable_objective_surface
    ),
    "B.reject.explicit_unknown_parameter_path": GateCaseImplementation(
        experiment_b_explicit_unknown_parameter_path
    ),
    "B.reject.overlapping_changed_unchanged_evidence": GateCaseImplementation(
        experiment_b_overlapping_changed_unchanged_evidence
    ),
    "B.reject.excluded_parameter_changes": GateCaseImplementation(
        experiment_b_excluded_parameter_changes
    ),
    "B.reject.excluded_optimizer_state_changes": GateCaseImplementation(
        experiment_b_excluded_optimizer_state_changes
    ),
    "B.reject.selected_zero_change_claimed_changed": GateCaseImplementation(
        experiment_b_selected_zero_change_claimed_changed
    ),
    "B.reject.unselected_parameter_claimed_changed": GateCaseImplementation(
        experiment_b_unselected_parameter_claimed_changed
    ),
}


__all__ = ["SECTION_IMPLEMENTATIONS"]
