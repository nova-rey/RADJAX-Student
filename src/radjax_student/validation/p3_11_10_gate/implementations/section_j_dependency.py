"""Section J real dependency-audit adversaries in isolated source trees."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_student.validation.architecture_audit import (
    build_architecture_audit,
    require_clean_architecture_audit,
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


def _audit(payload: Mapping[str, Any]) -> None:
    require_clean_architecture_audit(
        build_architecture_audit(Path(str(payload["root"])))
    )


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    root = context.temporary_root / case_id.replace(".", "_")
    package = root / "src" / "radjax_student" / "architecture"
    package.mkdir(parents=True)
    (root / "src" / "radjax_student" / "__init__.py").write_text("", encoding="utf-8")
    # Build the isolated fixture source without making this passive module an
    # optional-runtime importer itself.
    (package / "__init__.py").write_text("import " + "jax\n", encoding="utf-8")
    return prepare_public_input(
        baseline={"root": str(context.temporary_root), "edge": "none"},
        mutated={"root": str(root), "edge": f"architecture->{case_id}"},
        public_input_kind="source_tree",
        canonical_path="src/radjax_student/architecture/__init__.py",
        operation="insert_forbidden_import_edge",
        value_summary={"case": case_id},
        canonical_baseline={"root": "baseline", "edge": "none"},
        canonical_mutated={
            "root": "isolated-source-tree",
            "edge": f"architecture->{case_id}",
        },
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "dependency_audit_validation",
        _audit,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="J",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "J"
}
