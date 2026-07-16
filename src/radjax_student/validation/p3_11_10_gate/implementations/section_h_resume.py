"""Section H continuation evidence adversaries using strict replay parsing."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from radjax_student.validation.p3_11_9_replay.canonical import canonical_json_bytes
from radjax_student.validation.p3_11_9_replay.models import StatefulReplayReceipt
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    invoke_recorded_positive_control,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _parse_resume(payload: Mapping[str, Any]) -> None:
    StatefulReplayReceipt.from_json_bytes(canonical_json_bytes(payload))


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    return prepare_public_input(
        baseline={"schema_version": "radjax.p3_11_9_replay_evidence.v1"},
        mutated={"schema_version": f"invalid.resume.{case_id}"},
        public_input_kind="resume_lifecycle_evidence",
        canonical_path="schema_version",
        operation="replace_resume_schema",
        value_summary={"case": case_id},
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "resume_replay_validation",
        _parse_resume,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="H",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "H"
}
