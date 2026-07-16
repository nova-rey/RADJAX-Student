"""Literal Section C batch/objective binding adversaries."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from radjax_student.learning import LearningBatch
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    invoke_recorded_positive_control,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _validate_batch(payload: Mapping[str, Any]) -> None:
    LearningBatch(str(payload["batch_id"]), inputs={"x": float("nan")}, targets={})


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    return prepare_public_input(
        baseline={"batch_id": "valid", "inputs": {"x": 1.0}},
        mutated={"batch_id": case_id, "inputs": {"x": "nonfinite"}},
        public_input_kind="learning_batch",
        canonical_path="inputs.x",
        operation="replace_finite_input_with_nonfinite_input",
        value_summary={"case": case_id},
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "learning_batch_validation",
        _validate_batch,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="C",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "C"
}
