"""Architecture-independent learning contracts; P3.1 performs no learning."""

from radjax_student.learning.errors import (
    LEARNING_ERROR_CODES,
    LearningContractError,
    LearningErrorCode,
    LearningIssue,
)
from radjax_student.learning.models import (
    CHECKPOINT_POLICY_MODES,
    LEARNING_CLAIMS_NOT_MADE,
    LEARNING_STATE_SCHEMA_VERSION,
    METRIC_AGGREGATIONS,
    CheckpointPolicy,
    LearningBatch,
    LearningConfig,
    LearningReport,
    LearningState,
    LearningStepResult,
    LossResult,
    MetricRecord,
    canonical_learning_json,
)
from radjax_student.learning.protocols import ObjectiveEvaluator, UpdateScopeResolver
from radjax_student.learning.scopes import (
    OBJECTIVE_SCOPE_KINDS,
    UPDATE_SCOPE_KINDS,
    ObjectiveScope,
    ObjectiveScopeKind,
    ResolvedUpdateSelection,
    UpdateScope,
    UpdateScopeKind,
)

__all__ = [
    "CHECKPOINT_POLICY_MODES",
    "LEARNING_CLAIMS_NOT_MADE",
    "LEARNING_ERROR_CODES",
    "LEARNING_STATE_SCHEMA_VERSION",
    "METRIC_AGGREGATIONS",
    "OBJECTIVE_SCOPE_KINDS",
    "UPDATE_SCOPE_KINDS",
    "CheckpointPolicy",
    "LearningBatch",
    "LearningConfig",
    "LearningContractError",
    "LearningErrorCode",
    "LearningIssue",
    "LearningReport",
    "LearningState",
    "LearningStepResult",
    "LossResult",
    "MetricRecord",
    "ObjectiveEvaluator",
    "ObjectiveScope",
    "ObjectiveScopeKind",
    "ResolvedUpdateSelection",
    "UpdateScope",
    "UpdateScopeKind",
    "UpdateScopeResolver",
    "canonical_learning_json",
]
