"""The only P3.11.10B case-ID implementation registry.

Section modules own literal mutations.  This module performs only inventory and
identity validation; it never dispatches by broad execution class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from radjax_student.validation.p3_11_10_gate.inventory import CASES
from radjax_student.validation.p3_11_10_gate.models import GateCaseDefinition

from .common import (
    BoundaryProbe,
    GateCaseImplementation,
    GateExecutionContext,
    GateMutationDelta,
    PreparedGateCase,
    observe_failure,
)
from .section_a_contracts import (
    SECTION_IMPLEMENTATIONS as A,
)
from .section_b_layout import (
    SECTION_IMPLEMENTATIONS as B,
)
from .section_c_batch_objective import (
    SECTION_IMPLEMENTATIONS as C,
)
from .section_d_runtime import (
    SECTION_IMPLEMENTATIONS as D,
)
from .section_e_optimizer import (
    SECTION_IMPLEMENTATIONS as E,
)
from .section_f_loop import (
    SECTION_IMPLEMENTATIONS as F,
)
from .section_g_checkpoint import (
    SECTION_IMPLEMENTATIONS as G,
)
from .section_h_resume import (
    SECTION_IMPLEMENTATIONS as H,
)
from .section_i_replay import (
    SECTION_IMPLEMENTATIONS as I,
)
from .section_j_dependency import (
    SECTION_IMPLEMENTATIONS as J,
)
from .section_k_documentation import (
    SECTION_IMPLEMENTATIONS as K,
)

_SECTION_MAPPINGS = (A, B, C, D, E, F, G, H, I, J, K)
CASE_IMPLEMENTATIONS: Mapping[str, GateCaseImplementation] = {
    case_id: implementation
    for mapping in _SECTION_MAPPINGS
    for case_id, implementation in mapping.items()
}


def validate_implementations(
    cases: Iterable[GateCaseDefinition] = CASES,
) -> Mapping[str, GateCaseImplementation]:
    inventory = tuple(cases)
    expected = {case.case_id: case for case in inventory}
    actual = set(CASE_IMPLEMENTATIONS)
    if actual != set(expected):
        raise ValueError("p31110_case_implementation_incomplete")
    identities: set[str] = set()
    for case_id, definition in expected.items():
        implementation = CASE_IMPLEMENTATIONS[case_id]
        if (
            implementation.case_id != case_id
            or implementation.section_id != definition.section_id
        ):
            raise ValueError("p31110_case_implementation_incomplete")
        if implementation.execution_class != definition.execution_class:
            raise ValueError("p31110_case_implementation_incomplete")
        if implementation.expected_boundary != definition.boundary:
            raise ValueError("p31110_case_implementation_incomplete")
        if implementation.effective_identity in identities:
            raise ValueError("p31110_case_implementation_incomplete")
        identities.add(implementation.effective_identity)
    return CASE_IMPLEMENTATIONS


__all__ = [
    "BoundaryProbe",
    "CASE_IMPLEMENTATIONS",
    "GateCaseImplementation",
    "GateExecutionContext",
    "GateMutationDelta",
    "PreparedGateCase",
    "observe_failure",
    "validate_implementations",
]
