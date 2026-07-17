"""Dependency-free shared contracts for the pre-Phase-4 learning conveyor."""

from radjax_student.contracts.hf import HFPreservationReference
from radjax_student.contracts.layout import (
    JaxOptimizerStateDescriptor,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.contracts.metrics import METRIC_AGGREGATIONS, MetricRecord
from radjax_student.contracts.objective import (
    OBJECTIVE_CAPABILITY_SCHEMA_VERSION,
    OBJECTIVE_CONFIG_SCHEMA_VERSION,
    OBJECTIVE_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    ObjectiveCapabilityProfile,
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveExecutionDescriptor,
    ObjectiveIdentity,
    ResolvedObjectiveSelection,
    canonical_objective_json,
    objective_digest,
)
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
    "OBJECTIVE_CAPABILITY_SCHEMA_VERSION",
    "OBJECTIVE_CONFIG_SCHEMA_VERSION",
    "OBJECTIVE_EXECUTION_DESCRIPTOR_SCHEMA_VERSION",
    "ObjectiveCapabilityProfile",
    "ObjectiveConfig",
    "ObjectiveContractError",
    "ObjectiveExecutionDescriptor",
    "ObjectiveIdentity",
    "ObjectiveScope",
    "ObjectiveScopeKind",
    "ParameterTreeLayout",
    "ParameterTreeLayoutEntry",
    "ResolvedUpdateSelection",
    "ResolvedObjectiveSelection",
    "UpdateScope",
    "UpdateScopeKind",
    "canonical_objective_json",
    "objective_digest",
]
