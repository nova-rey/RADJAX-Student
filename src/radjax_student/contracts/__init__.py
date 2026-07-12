"""Dependency-free shared contracts for the pre-Phase-4 learning conveyor."""

from radjax_student.contracts.hf import HFPreservationReference
from radjax_student.contracts.layout import (
    JaxOptimizerStateDescriptor,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)

__all__ = [
    "HFPreservationReference",
    "JaxOptimizerStateDescriptor",
    "ParameterTreeLayout",
    "ParameterTreeLayoutEntry",
]
