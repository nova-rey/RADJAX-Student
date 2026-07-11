from __future__ import annotations

import pytest

from radjax_student.architecture import ArchitectureContractError, ForwardResult
from radjax_student.learning import ForwardObjectiveEvaluator


def test_forward_result_exposes_named_runtime_surfaces():
    result = ForwardResult(
        outputs=(1.0,),
        intermediate_surfaces=("hidden", "final_output"),
        surface_values={"hidden": (0.5,)},
    )

    assert result.surface("final_output") == (1.0,)
    assert result.surface("hidden") == (0.5,)
    assert result.to_dict()["surface_ids"] == ["hidden"]


def test_forward_result_rejects_missing_surface():
    with pytest.raises(ArchitectureContractError, match="unavailable"):
        ForwardResult(outputs=(1.0,)).surface("hidden")


def test_forward_objective_protocol_is_runtime_checkable_by_shape():
    class Objective:
        def evaluate(self, surface, targets, weights, objective_config):
            del targets, weights, objective_config
            return surface, {"surface_seen": True}

    assert isinstance(Objective(), ForwardObjectiveEvaluator)
