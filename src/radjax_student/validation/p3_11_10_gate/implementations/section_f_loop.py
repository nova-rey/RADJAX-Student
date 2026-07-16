"""Literal Section F generic-loop boundary adversaries."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    invoke_recorded_positive_control,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _construct_invalid_executor(payload: Mapping[str, Any]) -> None:
    from radjax_student.steps.jax_loop import JaxLoopExecutor

    JaxLoopExecutor(None, None, None, payload)  # type: ignore[arg-type]


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    return prepare_public_input(
        baseline={"executor": "complete", "event": "before_step"},
        mutated={"executor": "incomplete", "event": case_id},
        public_input_kind="loop_executor",
        canonical_path="executor.dependencies",
        operation="remove_loop_executor_dependency",
        value_summary={"case": case_id},
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "loop_executor_validation",
        _construct_invalid_executor,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="F",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "F"
}
