"""Architecture-resolved execution planning for the optional JAX path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from radjax_student.architecture import (
    ArchitectureContractError,
    JaxArchitecturePlugin,
)
from radjax_student.contracts import (
    ObjectiveScope,
    ParameterTreeLayout,
    ResolvedObjectiveSelection,
    ResolvedUpdateSelection,
    UpdateScope,
)


@dataclass(frozen=True)
class JaxExecutionPlan:
    """Architecture-owned selections and the layout-derived update mask."""

    objective_selection: ResolvedObjectiveSelection
    update_selection: ResolvedUpdateSelection
    parameter_layout: ParameterTreeLayout
    update_mask: Any


def prepare_jax_execution_plan(
    *,
    architecture: JaxArchitecturePlugin,
    parameters: Any,
    parameter_layout: ParameterTreeLayout,
    objective_scope: ObjectiveScope,
    update_scope: UpdateScope,
) -> JaxExecutionPlan:
    """Resolve generic intent through the architecture before JAX executes."""

    if not isinstance(architecture, JaxArchitecturePlugin):
        raise TypeError("JAX execution requires a complete JaxArchitecturePlugin")
    if parameter_layout.architecture_id != architecture.architecture_id:
        raise ArchitectureContractError(
            "architecture_parameter_catalog_invalid",
            "parameter layout does not belong to the architecture plugin",
        )
    catalog = architecture.describe_parameters(parameters)
    if catalog.architecture_id != architecture.architecture_id:
        raise ArchitectureContractError(
            "architecture_parameter_catalog_invalid",
            "architecture returned a catalog for a different plugin",
        )
    if set(catalog.paths) != set(parameter_layout.logical_paths):
        raise ArchitectureContractError(
            "architecture_parameter_catalog_invalid",
            "parameter layout and architecture catalog paths differ",
            details={
                "catalog_paths": list(catalog.paths),
                "layout_paths": list(parameter_layout.logical_paths),
            },
        )
    parameter_layout.validate_materialized_parameters(parameters)
    for entry in parameter_layout.entries:
        descriptor = catalog.get(entry.logical_path)
        if (
            descriptor.shape != entry.shape
            or descriptor.dtype != entry.dtype
            or descriptor.trainable_by_default != entry.trainable
        ):
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "parameter layout does not match architecture parameter metadata",
                details={"logical_path": entry.logical_path},
            )
    objective_selection = architecture.resolve_objective_scope(
        objective_scope, architecture.architecture_metadata()
    )
    update_selection = architecture.resolve_update_scope(update_scope, catalog)
    return JaxExecutionPlan(
        objective_selection=objective_selection,
        update_selection=update_selection,
        parameter_layout=parameter_layout,
        update_mask=parameter_layout.update_mask(
            parameters, update_selection.selected_parameter_paths
        ),
    )


__all__ = ["JaxExecutionPlan", "prepare_jax_execution_plan"]
