from __future__ import annotations

import json

import pytest

from radjax_student.runtime import (
    PLACEMENT_CAPABILITY_MAPPING,
    PLACEMENT_INTENTS,
    LogicalAxisSpec,
    PlacementContractError,
    PlacementPlan,
    RuntimeConfig,
    ValuePlacementSpec,
    placement_capabilities,
    unresolved_placement_resolution,
)


@pytest.mark.parametrize("intent", PLACEMENT_INTENTS)
def test_each_public_placement_intent_is_serializable(intent: str) -> None:
    value = ValuePlacementSpec(
        value_path="runtime.scalar",
        placement=intent,
        logical_axes=(),
    )

    assert value.placement == intent
    assert value.required_capabilities == placement_capabilities(intent)


def test_capability_mapping_is_centralized_and_deterministic() -> None:
    assert placement_capabilities("single_device") == ("placement.single_device_v1",)
    assert placement_capabilities("replicated") == ("placement.replicated_v1",)
    assert placement_capabilities("data_sharded") == ("placement.data_sharded_v1",)
    assert placement_capabilities("model_sharded") == ("placement.model_sharded_v1",)
    assert placement_capabilities("automatic") == ()
    assert dict(PLACEMENT_CAPABILITY_MAPPING)["unspecified"] == ()


def test_plan_validates_paths_axes_and_contradictory_constraints() -> None:
    batch = LogicalAxisSpec("batch", size=8, sharding_role="data")
    model = LogicalAxisSpec("model", size=16, sharding_role="model")

    with pytest.raises(PlacementContractError) as duplicate_paths:
        PlacementPlan(
            plan_id="duplicate-paths",
            values=(
                ValuePlacementSpec("batch.tokens", "data_sharded", ("batch",)),
                ValuePlacementSpec("batch.tokens", "data_sharded", ("batch",)),
            ),
            logical_axis_catalog=(batch,),
        )
    assert duplicate_paths.value.code == "placement_value_path_duplicate"

    with pytest.raises(PlacementContractError) as duplicate_axes:
        PlacementPlan(
            plan_id="duplicate-axes",
            values=(),
            logical_axis_catalog=(batch, batch),
        )
    assert duplicate_axes.value.code == "placement_axis_duplicate"

    with pytest.raises(PlacementContractError) as unknown_axis:
        PlacementPlan(
            plan_id="unknown-axis",
            values=(ValuePlacementSpec("batch.tokens", "data_sharded", ("batch",)),),
        )
    assert unknown_axis.value.code == "placement_axis_unknown"

    with pytest.raises(PlacementContractError) as missing_role:
        PlacementPlan(
            plan_id="missing-role",
            values=(ValuePlacementSpec("params.weight", "data_sharded", ("model",)),),
            logical_axis_catalog=(model,),
        )
    assert missing_role.value.code == "placement_constraint_conflict"

    with pytest.raises(PlacementContractError) as conflict:
        PlacementPlan(
            plan_id="replica-conflict",
            values=(
                ValuePlacementSpec(
                    "runtime.scalar",
                    "replicated",
                    constraints=("must_be_divisible_by_device_count",),
                ),
            ),
        )
    assert conflict.value.code == "placement_constraint_conflict"


def test_unspecified_rejects_requirements_and_differs_from_automatic() -> None:
    with pytest.raises(PlacementContractError) as invalid:
        ValuePlacementSpec(
            "runtime.scalar",
            "unspecified",
            required_capabilities=("placement.single_device_v1",),
        )

    automatic = unresolved_placement_resolution("automatic")
    unspecified = unresolved_placement_resolution("unspecified")

    assert invalid.value.code == "placement_constraint_conflict"
    assert automatic.status == unspecified.status == "unresolved"
    assert automatic.intent != unspecified.intent
    assert automatic.warnings[0].code == "placement_automatic_unresolved"
    assert unspecified.warnings[0].code == "placement_unspecified"


def test_plugin_axis_and_unknown_size_are_preserved_as_warnings() -> None:
    plan = PlacementPlan(
        plan_id="plugin-axis",
        values=(
            ValuePlacementSpec("params.adapter.weight", "model_sharded", ("adapter",)),
        ),
        logical_axis_catalog=(LogicalAxisSpec("adapter", sharding_role="model"),),
    )

    codes = [item.code for item in plan.warnings]
    assert "placement_plugin_defined_axis" in codes
    assert "placement_axis_size_unknown" in codes
    assert plan.logical_axis_catalog[0].name == "adapter"


def test_plan_round_trip_and_precedence_are_deterministic() -> None:
    plan = PlacementPlan(
        plan_id="precedence",
        values=(
            ValuePlacementSpec("batch.tokens", "unspecified", ("batch",)),
            ValuePlacementSpec("runtime.scalar", "single_device"),
        ),
        logical_axis_catalog=(LogicalAxisSpec("batch", sharding_role="data"),),
        default_placement="replicated",
    )
    payload = plan.to_dict()

    assert PlacementPlan.from_dict(payload) == plan
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert (
        plan.effective_placement("runtime.scalar", RuntimeConfig()) == "single_device"
    )
    assert plan.effective_placement("batch.tokens", RuntimeConfig()) == "replicated"

    config_only = PlacementPlan(
        plan_id="config-only",
        values=(ValuePlacementSpec("batch.tokens", "unspecified", ("batch",)),),
        logical_axis_catalog=(LogicalAxisSpec("batch", sharding_role="data"),),
    )
    assert (
        config_only.effective_placement(
            "batch.tokens",
            RuntimeConfig(placement_policy="data_sharded"),
        )
        == "data_sharded"
    )
    assert (
        config_only.effective_placement("batch.tokens", RuntimeConfig())
        == "unspecified"
    )


def test_placement_module_has_no_execution_or_architecture_dependencies() -> None:
    source = (
        _repo_root() / "src" / "radjax_student" / "runtime" / "placement.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "import jax",
        "import numpy",
        "radjax_student.architecture",
        "radjax_student.students",
        "radjax_student.training",
        "radjax_student.artifacts",
        "device_put",
        "NamedSharding",
        "PartitionSpec",
        "Mesh(",
        "socket",
        "urllib",
    ):
        assert forbidden not in source


def _repo_root():
    return __import__("pathlib").Path(__file__).resolve().parents[1]
