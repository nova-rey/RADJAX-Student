"""Section G checkpoint adversaries through the public v3 restore API."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_student.checkpoints import load_learning_checkpoint_v3
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    invoke_recorded_positive_control,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _restore(payload: Mapping[str, Any]) -> None:
    directory = Path(str(payload["directory"]))
    load_learning_checkpoint_v3(
        directory, optimizer=object(), parameter_layout=object()
    )  # type: ignore[arg-type]


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    baseline = context.temporary_root / "valid-baseline"
    baseline.mkdir()
    mutated = context.temporary_root / f"mutated-{case_id.rsplit('.', 1)[-1]}"
    mutated.mkdir()
    return prepare_public_input(
        baseline={"directory": str(baseline), "manifest": "present"},
        mutated={"directory": str(mutated), "manifest": "missing", "mutation": case_id},
        public_input_kind="checkpoint_directory",
        canonical_path="manifest.json",
        operation="remove_manifest_file",
        value_summary={"case": case_id},
        canonical_baseline={"directory": "valid-baseline", "manifest": "present"},
        canonical_mutated={
            "directory": "mutated",
            "manifest": "missing",
            "mutation": case_id,
        },
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "checkpoint_restore_validation",
        _restore,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="G",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "G"
}
