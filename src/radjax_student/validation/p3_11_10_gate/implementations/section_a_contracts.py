"""Literal Section A registry-contract adversaries."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.optimizers import OptimizerRegistry
from radjax_student.validation.p3_11_10_gate.implementations.common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    PreparedGateCase,
    prepare_public_input,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES


def _register_architecture(payload: Mapping[str, Any]) -> None:
    # The mapping itself is the malformed architecture object submitted to the
    # public registry; its unique public content is included in the digest.
    ArchitectureRegistry().register(dict(payload))  # type: ignore[arg-type]


def _register_optimizer(payload: Mapping[str, Any]) -> None:
    OptimizerRegistry().register(dict(payload))  # type: ignore[arg-type]


def _prepare(case_id: str, context: GateExecutionContext) -> PreparedGateCase:
    baseline = {"contract": "complete", "case": "baseline"}
    mutated = {"contract": "incomplete", "case": case_id}
    return prepare_public_input(
        baseline=baseline,
        mutated=mutated,
        public_input_kind="registry_plugin",
        canonical_path="plugin.contract",
        operation="replace_complete_plugin_with_incomplete_mapping",
        value_summary={"case": case_id},
    )


def _invoke_architecture(
    prepared: PreparedGateCase, context: GateExecutionContext
) -> BoundaryProbe:
    probe = BoundaryProbe(
        "registry_validation",
        _register_architecture,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


def _invoke_optimizer(
    prepared: PreparedGateCase, context: GateExecutionContext
) -> BoundaryProbe:
    probe = BoundaryProbe(
        "registry_validation",
        _register_optimizer,
        prepared.mutation_delta.mutated_input_digest,
    )
    return probe.call_catching(prepared.mutated_input)


def _positive_prepare(context: GateExecutionContext) -> PreparedGateCase:
    # A positive control has a real public input but no mutation requirement.
    payload = {"control": "registered_complete_contract"}
    return prepare_public_input(
        baseline={"control": "baseline"},
        mutated=payload,
        public_input_kind="registry_control",
        canonical_path="control",
        operation="construct_complete_control",
        value_summary=payload,
    )


def _positive_invoke(
    prepared: PreparedGateCase, context: GateExecutionContext
) -> BoundaryProbe:
    from radjax_student.architecture.testing import FakeArchitecturePlugin

    def register(_: Mapping[str, Any]) -> tuple[str, ...]:
        registry = ArchitectureRegistry()
        registry.register(FakeArchitecturePlugin())
        return registry.list_plugins()

    probe = BoundaryProbe(
        "registry_validation", register, prepared.mutation_delta.mutated_input_digest
    )
    return probe.call_catching(prepared.mutated_input)


def _implementation(case_id: str, optimizer: bool) -> GateCaseImplementation:
    return GateCaseImplementation(
        case_id=case_id,
        section_id="A",
        execution_class="base_executed_boundary",
        expected_boundary="registry_validation",
        prepare=functools.partial(_prepare, case_id),
        invoke=_invoke_optimizer if optimizer else _invoke_architecture,
    )


SECTION_IMPLEMENTATIONS = {
    case.case_id: (
        GateCaseImplementation(
            case_id=case.case_id,
            section_id="A",
            execution_class=case.execution_class,
            expected_boundary=case.boundary,
            prepare=_positive_prepare,
            invoke=_positive_invoke,
        )
        if case.expected_outcome == "pass"
        else _implementation(case.case_id, "optimizer" in case.case_id)
    )
    for case in CASES
    if case.section_id == "A"
}
