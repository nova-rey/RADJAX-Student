"""The literal P3.11.10C function registry.

Section modules export explicit ``experiment_*`` functions.  This module is
the sole inventory-aware surface: it binds those functions to ordered case IDs
and rejects any missing, duplicate, generalized, or undeclared implementation
before the gate starts executing public boundaries.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping

from radjax_student.validation.p3_11_10_gate.inventory import CASES
from radjax_student.validation.p3_11_10_gate.models import GateCaseDefinition

from .common import (
    BoundaryProbe,
    ExperimentExecution,
    GateCaseImplementation,
    GateExecutionContext,
    MutationDelta,
    observe_failure,
)
from .section_a_contracts import SECTION_IMPLEMENTATIONS as A
from .section_b_layout import SECTION_IMPLEMENTATIONS as B
from .section_c_batch_objective import SECTION_IMPLEMENTATIONS as C
from .section_d_runtime import SECTION_IMPLEMENTATIONS as D
from .section_e_optimizer import SECTION_IMPLEMENTATIONS as E
from .section_f_loop import SECTION_IMPLEMENTATIONS as F
from .section_g_checkpoint import SECTION_IMPLEMENTATIONS as G
from .section_h_resume import SECTION_IMPLEMENTATIONS as H
from .section_i_replay import SECTION_IMPLEMENTATIONS as I
from .section_j_dependency import SECTION_IMPLEMENTATIONS as J
from .section_k_documentation import SECTION_IMPLEMENTATIONS as K

_SECTION_MAPPINGS: tuple[tuple[str, Mapping[str, GateCaseImplementation]], ...] = (
    ("A", A),
    ("B", B),
    ("C", C),
    ("D", D),
    ("E", E),
    ("F", F),
    ("G", G),
    ("H", H),
    ("I", I),
    ("J", J),
    ("K", K),
)
_RAW_CASE_IMPLEMENTATIONS: dict[str, GateCaseImplementation] = {}
for _section, _mapping in _SECTION_MAPPINGS:
    overlap = set(_RAW_CASE_IMPLEMENTATIONS) & set(_mapping)
    if overlap:
        raise RuntimeError("p31110_case_implementation_duplicate")
    _RAW_CASE_IMPLEMENTATIONS.update(_mapping)

_DEFINITIONS_BY_ID = {definition.case_id: definition for definition in CASES}
CASE_IMPLEMENTATIONS: dict[str, GateCaseImplementation] = {
    identifier: implementation.bind(
        case_id=_DEFINITIONS_BY_ID[identifier].case_id,
        expected_boundary=_DEFINITIONS_BY_ID[identifier].boundary,
        execution_class=_DEFINITIONS_BY_ID[identifier].execution_class,
    )
    for identifier, implementation in _RAW_CASE_IMPLEMENTATIONS.items()
    if identifier in _DEFINITIONS_BY_ID
}

_FORBIDDEN_IMPLEMENTATION_TOKENS = (
    "case_id",
    "expected_failure",
    "expected_boundary",
    "expected_outcome",
)


def implementation_identity_for(definition: GateCaseDefinition) -> str:
    """Bind an actual literal function to its inventory-owned declaration.

    The inventory supplies expectation metadata only here, after an experiment
    was selected.  Literal functions and observers never receive it.
    """

    implementation = CASE_IMPLEMENTATIONS[definition.case_id]
    if (
        implementation.case_id != definition.case_id
        or implementation.expected_boundary != definition.boundary
        or implementation.execution_class != definition.execution_class
    ):
        raise ValueError("p31110_case_implementation_binding_mismatch")
    return implementation.identity


def _function_source(implementation: GateCaseImplementation) -> str:
    try:
        return inspect.getsource(implementation.function)
    except (OSError, TypeError) as exc:  # pragma: no cover - maintainer error
        raise ValueError("p31110_case_implementation_source_unavailable") from exc


def validate_implementations(
    cases: Iterable[GateCaseDefinition] = CASES,
) -> Mapping[str, GateCaseImplementation]:
    """Prove inventory coverage and one literal source function per entry."""

    inventory = tuple(cases)
    expected = {item.case_id: item for item in inventory}
    if set(_RAW_CASE_IMPLEMENTATIONS) != set(expected):
        raise ValueError("p31110_case_implementation_raw_inventory_mismatch")
    actual_ids = set(CASE_IMPLEMENTATIONS)
    if actual_ids != set(expected):
        raise ValueError("p31110_case_implementation_incomplete")
    functions: set[str] = set()
    behavior_identities: set[str] = set()
    identities: set[str] = set()
    for definition in inventory:
        implementation = CASE_IMPLEMENTATIONS[definition.case_id]
        function = implementation.function
        if not function.__name__.startswith("experiment_"):
            raise ValueError("p31110_case_implementation_not_literal")
        source = _function_source(implementation)
        if any(token in source for token in _FORBIDDEN_IMPLEMENTATION_TOKENS):
            raise ValueError("p31110_case_implementation_uses_inventory_metadata")
        signature = inspect.signature(function)
        if tuple(signature.parameters) != ("context",):
            raise ValueError("p31110_case_implementation_requires_variant_arguments")
        if function.__qualname__ in functions:
            raise ValueError("p31110_case_implementation_function_reused")
        if implementation.case_id != definition.case_id:
            raise ValueError("p31110_case_implementation_case_binding_mismatch")
        if implementation.expected_boundary != definition.boundary:
            raise ValueError("p31110_case_implementation_boundary_mismatch")
        if implementation.execution_class != definition.execution_class:
            raise ValueError("p31110_case_implementation_execution_class_mismatch")
        if not all(
            (
                implementation.preparation_identity,
                implementation.mutation_identity,
                implementation.invocation_identity,
                implementation.observation_adapter_identity,
            )
        ):
            raise ValueError("p31110_case_implementation_identity_incomplete")
        if implementation.behavior_identity in behavior_identities:
            raise ValueError("p31110_case_implementation_behavior_reused")
        if implementation.identity in identities:
            raise ValueError("p31110_case_implementation_identity_reused")
        functions.add(function.__qualname__)
        behavior_identities.add(implementation.behavior_identity)
        identities.add(implementation.identity)
    for section, mapping in _SECTION_MAPPINGS:
        if any(not identifier.startswith(f"{section}.") for identifier in mapping):
            raise ValueError("p31110_case_implementation_wrong_section")
    return CASE_IMPLEMENTATIONS


__all__ = [
    "BoundaryProbe",
    "CASE_IMPLEMENTATIONS",
    "ExperimentExecution",
    "GateCaseImplementation",
    "GateExecutionContext",
    "MutationDelta",
    "implementation_identity_for",
    "observe_failure",
    "validate_implementations",
]
