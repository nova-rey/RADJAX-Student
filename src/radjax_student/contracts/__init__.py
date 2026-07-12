"""Dependency-free shared contracts for the pre-Phase-4 learning conveyor."""

from radjax_student.contracts.hf import HFPreservationReference
from radjax_student.contracts.layout import (
    JaxOptimizerStateDescriptor,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.contracts.metrics import METRIC_AGGREGATIONS, MetricRecord
from radjax_student.contracts.objective import ResolvedObjectiveSelection
from radjax_student.contracts.shared import (
    LearningBatch,
    ObjectiveScope,
    ObjectiveScopeKind,
    ResolvedUpdateSelection,
    UpdateScope,
    UpdateScopeKind,
)

__all__ = [
    "HFPreservationReference",
    "JaxOptimizerStateDescriptor",
    "LearningBatch",
    "MetricRecord",
    "METRIC_AGGREGATIONS",
    "ObjectiveScope",
    "ObjectiveScopeKind",
    "ParameterTreeLayout",
    "ParameterTreeLayoutEntry",
    "ResolvedUpdateSelection",
    "ResolvedObjectiveSelection",
    "UpdateScope",
    "UpdateScopeKind",
]
