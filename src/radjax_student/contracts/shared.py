"""Compatibility-owned shared vocabulary during the P3.11 migration."""

# These aliases preserve object identity while public imports migrate. The
# canonical implementations remain dependency-free from architecture/optimizer.
from radjax_student.contracts.batch import LearningBatch
from radjax_student.contracts.metrics import MetricRecord
from radjax_student.contracts.scopes import (
    ObjectiveScope,
    ObjectiveScopeKind,
    ResolvedUpdateSelection,
    UpdateScope,
    UpdateScopeKind,
)

__all__ = [
    "LearningBatch",
    "MetricRecord",
    "ObjectiveScope",
    "ObjectiveScopeKind",
    "ResolvedUpdateSelection",
    "UpdateScope",
    "UpdateScopeKind",
]
