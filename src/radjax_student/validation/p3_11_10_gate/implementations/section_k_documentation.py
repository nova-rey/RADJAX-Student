"""Section K real documentation-claim audit adversaries."""

from __future__ import annotations

import functools
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_student.validation.p3_11_10_gate.documentation import (
    check_closure_documentation,
    maintained_paths,
    require_closure_documentation,
)
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    invoke_recorded_positive_control,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _validate(payload: Mapping[str, Any]) -> None:
    require_closure_documentation(
        check_closure_documentation(Path(str(payload["root"])))
    )


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    root = context.temporary_root / case_id.replace(".", "_")
    for relative in maintained_paths():
        source = context.repository_root / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    target = root / maintained_paths()[0]
    target.write_text(
        target.read_text(encoding="utf-8") + f"\nremote CI passed {case_id}\n",
        encoding="utf-8",
    )
    return prepare_public_input(
        baseline={"root": str(context.repository_root), "claim": "accepted"},
        mutated={"root": str(root), "claim": f"remote-ci-passed:{case_id}"},
        public_input_kind="maintained_documents",
        canonical_path=maintained_paths()[0],
        operation="insert_stale_status_claim",
        value_summary={"case": case_id},
        canonical_baseline={"root": "maintained-documents", "claim": "accepted"},
        canonical_mutated={
            "root": "isolated-documents",
            "claim": f"remote-ci-passed:{case_id}",
        },
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "documentation_validation",
        _validate,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="K",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "K"
}
