"""Architecture-independent learning contracts; P3.1 performs no learning."""

from radjax_student.learning.errors import (
    LEARNING_ERROR_CODES,
    LearningContractError,
    LearningErrorCode,
    LearningIssue,
)
from radjax_student.learning.hooks import (
    HookContext,
    HookPolicy,
    HookResult,
    dispatch_hooks,
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
from radjax_student.learning.objectives import (
    BATCH_OBJECTIVE_CLAIMS_NOT_MADE,
    WEIGHTING_POLICIES,
    BatchMetadata,
    ObjectiveRequest,
    ObjectiveResult,
    WeightingPolicy,
    canonical_objective_json,
)
from radjax_student.learning.protocols import ObjectiveEvaluator, UpdateScopeResolver
from radjax_student.learning.run_report import (
    LearningRunReport,
    RunCheckpointSummary,
    RunIssueSummary,
    RunLifecycleSummary,
    RunMetricSummary,
    RunScopeSummary,
    RunStatusSummary,
    build_learning_run_report,
)
from radjax_student.learning.scopes import (
    OBJECTIVE_SCOPE_KINDS,
    UPDATE_SCOPE_KINDS,
    ObjectiveScope,
    ObjectiveScopeKind,
    ResolvedUpdateSelection,
    UpdateScope,
    UpdateScopeKind,
)
from radjax_student.learning.telemetry import (
    LearningEvent,
    MetricRetentionPolicy,
    MetricSeries,
    MetricSummary,
)

__all__ = [
    "CHECKPOINT_POLICY_MODES",
    "BATCH_OBJECTIVE_CLAIMS_NOT_MADE",
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
    "MetricRetentionPolicy",
    "MetricSeries",
    "MetricSummary",
    "LearningEvent",
    "LearningRunReport",
    "RunCheckpointSummary",
    "RunIssueSummary",
    "RunLifecycleSummary",
    "RunMetricSummary",
    "RunScopeSummary",
    "RunStatusSummary",
    "build_learning_run_report",
    "HookContext",
    "HookPolicy",
    "HookResult",
    "dispatch_hooks",
    "BatchMetadata",
    "ObjectiveRequest",
    "ObjectiveResult",
    "ObjectiveEvaluator",
    "ObjectiveScope",
    "ObjectiveScopeKind",
    "P38ObservabilityAcceptanceReceipt",
    "ResolvedUpdateSelection",
    "UpdateScope",
    "UpdateScopeKind",
    "UpdateScopeResolver",
    "WEIGHTING_POLICIES",
    "WeightingPolicy",
    "canonical_learning_json",
    "canonical_objective_json",
    "run_p3_8_observability_acceptance",
    "P39SyntheticLearningReceipt",
    "SyntheticRunSummary",
    "run_p3_9_synthetic_learning_smoke",
]


def __getattr__(name):
    if name in {
        "P38ObservabilityAcceptanceReceipt",
        "run_p3_8_observability_acceptance",
    }:
        from radjax_student.learning.observability_acceptance import (
            P38ObservabilityAcceptanceReceipt,
            run_p3_8_observability_acceptance,
        )

        return {
            "P38ObservabilityAcceptanceReceipt": P38ObservabilityAcceptanceReceipt,
            "run_p3_8_observability_acceptance": run_p3_8_observability_acceptance,
        }[name]
    if name in {
        "P39SyntheticLearningReceipt",
        "SyntheticRunSummary",
        "run_p3_9_synthetic_learning_smoke",
    }:
        from radjax_student.learning.synthetic_smoke import (
            P39SyntheticLearningReceipt,
            SyntheticRunSummary,
            run_p3_9_synthetic_learning_smoke,
        )

        return {
            "P39SyntheticLearningReceipt": P39SyntheticLearningReceipt,
            "SyntheticRunSummary": SyntheticRunSummary,
            "run_p3_9_synthetic_learning_smoke": run_p3_9_synthetic_learning_smoke,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
