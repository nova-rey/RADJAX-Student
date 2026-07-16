"""Literal Section E optimizer contract adversaries."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from radjax_student.optimizers import OptimizerRegistry
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    invoke_recorded_positive_control,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _validate_optimizer(payload: Mapping[str, Any]) -> None:
    OptimizerRegistry().register(dict(payload))  # type: ignore[arg-type]


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    return prepare_public_input(
        baseline={"optimizer_id": "sgd", "capability": "complete"},
        mutated={"optimizer_id": f"broken.{case_id}", "capability": "missing"},
        public_input_kind="optimizer_backend",
        canonical_path="capability",
        operation="remove_optimizer_backend_contract",
        value_summary={"case": case_id},
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "optimizer_registry_validation",
        _validate_optimizer,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="E",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "E"
}
