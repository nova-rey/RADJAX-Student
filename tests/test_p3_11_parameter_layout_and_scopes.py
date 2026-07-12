"""P3.11.3 proves layout-driven masks and architecture-owned selections."""

from __future__ import annotations

from typing import Any

import pytest

from radjax_student.architecture import (
    ArchitectureCapabilityProfile,
    ArchitectureContractError,
    ForwardResult,
    JaxArchitecturePlugin,
)
from radjax_student.architecture.testing import (
    FAKE_ARCHITECTURE_CAPABILITIES,
    FakeArchitecturePlugin,
)
from radjax_student.contracts import ParameterTreeLayout, ParameterTreeLayoutEntry
from radjax_student.learning import ObjectiveScope, UpdateScope
from radjax_student.learning.jax_execution import prepare_jax_execution_plan


class CompleteJaxArchitecture(FakeArchitecturePlugin):
    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return ArchitectureCapabilityProfile(
            architecture_id=self.architecture_id,
            version=self.architecture_version,
            capabilities=(
                *FAKE_ARCHITECTURE_CAPABILITIES,
                "architecture.jax_execution_v1",
            ),
        )

    def apply_jax(
        self,
        parameters: Any,
        architecture_state: Any,
        batch: Any,
        *,
        objective_scope: ObjectiveScope,
        training: bool,
        rng_key: Any | None,
    ) -> ForwardResult:
        del parameters, architecture_state, batch, objective_scope, training, rng_key
        return ForwardResult(outputs="final", surface_values={"trunk_output": "trunk"})


def _layout() -> ParameterTreeLayout:
    return ParameterTreeLayout(
        architecture_id="test.architecture.v1",
        entries=(
            ParameterTreeLayoutEntry(
                "trunk.weight",
                ("trunk", "weight"),
                (4, 4),
                "float32",
                "recurrent_block",
                ("trunk", "shared", "whole_student"),
            ),
            ParameterTreeLayoutEntry(
                "trunk.bias",
                ("trunk", "bias"),
                (4,),
                "float32",
                "recurrent_block",
                ("trunk", "whole_student"),
            ),
            ParameterTreeLayoutEntry(
                "head.weight",
                ("head", "weight"),
                (4, 4),
                "float32",
                "output_head",
                ("head", "shared", "whole_student"),
            ),
        ),
    )


def _parameters() -> dict[str, dict[str, object]]:
    class Leaf:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = "float32"

    return {
        "trunk": {"weight": Leaf((4, 4)), "bias": Leaf((4,))},
        "head": {"weight": Leaf((4, 4))},
    }


def test_complete_jax_plugin_resolves_scopes_and_layout_drives_mask():
    architecture = CompleteJaxArchitecture()
    assert isinstance(architecture, JaxArchitecturePlugin)
    plan = prepare_jax_execution_plan(
        architecture=architecture,
        parameters=_parameters(),
        parameter_layout=_layout(),
        objective_scope=ObjectiveScope("intermediate_surface", "trunk_output"),
        update_scope=UpdateScope("named_region", "trunk"),
    )
    assert plan.objective_selection.surface_id == "trunk_output"
    assert plan.update_selection.selected_parameter_paths == (
        "trunk.bias",
        "trunk.weight",
    )
    assert plan.update_mask == {
        "head": {"weight": False},
        "trunk": {"bias": True, "weight": True},
    }


def test_layout_rejects_extra_or_stale_pytree_leaves():
    with pytest.raises(ValueError, match="keys"):
        _layout().update_mask(
            {
                "trunk": {"weight": object(), "bias": object(), "extra": object()},
                "head": {"weight": object()},
            },
            ("trunk.weight",),
        )


def test_forward_result_requires_resolved_selection_not_raw_scope():
    with pytest.raises(ArchitectureContractError, match="resolved objective"):
        ForwardResult(outputs="final").surface_for(ObjectiveScope())


def test_execution_plan_rejects_layout_catalog_mismatch():
    with pytest.raises(
        ArchitectureContractError, match="layout and architecture catalog"
    ):
        prepare_jax_execution_plan(
            architecture=CompleteJaxArchitecture(),
            parameters=_parameters(),
            parameter_layout=ParameterTreeLayout(
                architecture_id="test.architecture.v1",
                entries=_layout().entries[:-1],
            ),
            objective_scope=ObjectiveScope(),
            update_scope=UpdateScope(),
        )
