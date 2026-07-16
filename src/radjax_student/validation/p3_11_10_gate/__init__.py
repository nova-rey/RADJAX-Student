"""Passive contracts for the P3.11.10 final adversarial gate.

Importing this package deliberately does not import JAX.  The executable
conveyor is loaded only by the gate runner after command-line parsing.
"""

from radjax_student.validation.p3_11_10_gate.inventory import CASES, SECTIONS
from radjax_student.validation.p3_11_10_gate.models import (
    FinalAdversarialGateReceipt,
    GateBlocker,
    GateCaseDefinition,
    GateCaseResult,
)

__all__ = [
    "CASES",
    "SECTIONS",
    "FinalAdversarialGateReceipt",
    "GateBlocker",
    "GateCaseDefinition",
    "GateCaseResult",
]
