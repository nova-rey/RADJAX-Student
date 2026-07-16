"""Literal Section B parameter-layout and scope adversaries."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from radjax_student.contracts import ParameterTreeLayout
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    invoke_recorded_positive_control,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _validate_layout(payload: Mapping[str, Any]) -> None:
    ParameterTreeLayout(str(payload["architecture_id"]), ())


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    return prepare_public_input(
        baseline={"architecture_id": "test.architecture.v1", "entries": 1},
        mutated={"architecture_id": f"test.architecture.{case_id}", "entries": 0},
        public_input_kind="parameter_tree_layout",
        canonical_path="entries",
        operation="remove_required_layout_entry",
        value_summary={"case": case_id},
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "parameter_layout_validation",
        _validate_layout,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


def _positive(
    prepared: PreparedGateCase, context: GateExecutionContext
) -> BoundaryProbe:
    def validate(_: Mapping[str, Any]) -> str:
        return "layout_control_recorded"

    probe = BoundaryProbe(
        "parameter_layout_validation",
        validate,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="B",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "B"
}
