"""Section D runtime ownership adversaries, with JAX imported only on invoke."""

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


def _derive_invalid_key(payload: Mapping[str, Any]) -> None:
    from radjax_student.runtime import RuntimeKeys
    from radjax_student.runtime.jax_bridge import derive_jax_key

    derive_jax_key(
        RuntimeKeys.from_seed(int(payload["seed"])).dropout,
        global_step=int(payload["global_step"]),
        micro_step=int(payload["micro_step"]),
        slot=str(payload["slot"]),
        invocation_index=int(payload["invocation_index"]),
    )


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    return prepare_public_input(
        baseline={
            "seed": 17,
            "global_step": 0,
            "micro_step": 0,
            "slot": "dropout",
            "invocation_index": 0,
        },
        mutated={
            "seed": 17,
            "global_step": 0,
            "micro_step": 0,
            "slot": f"invalid_slot_{case_id}",
            "invocation_index": 0,
        },
        public_input_kind="runtime_rng_request",
        canonical_path="slot",
        operation="replace_rng_slot",
        value_summary={"case": case_id},
    )


def _invoke(prepared: PreparedGateCase, context: GateExecutionContext) -> BoundaryProbe:
    probe = BoundaryProbe(
        "runtime_rng_validation",
        _derive_invalid_key,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


SECTION_IMPLEMENTATIONS = {
    case.case_id: GateCaseImplementation(
        case_id=case.case_id,
        section_id="D",
        execution_class=case.execution_class,
        expected_boundary=case.boundary,
        prepare=functools.partial(_prepare, case.case_id),
        invoke=invoke_recorded_positive_control
        if case.expected_outcome == "pass"
        else _invoke,
    )
    for case in CASES
    if case.section_id == "D"
}
