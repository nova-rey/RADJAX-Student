"""Layered learning checkpoint contract."""

from radjax_student.checkpoints.learning import (
    CONTINUATION_CHECKPOINT_ROLE,
    HF_DISTRIBUTION_CHECKPOINT_ROLE,
    LearningCheckpoint,
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from radjax_student.checkpoints.roles import (
    CheckpointPayloadDescriptor,
    FutureTensorPayloadDescriptor,
    reject_implicit_hf_conversion,
)
from radjax_student.checkpoints.v3 import (
    CHECKPOINT_OBJECTIVE_IDENTITY_MISSING,
    CHECKPOINT_OPTIMIZER_STEP_MISMATCH,
    CHECKPOINT_V3_SCHEMA_VERSION,
    HISTORICAL_MSE_OBJECTIVE_ALIASES,
    CheckpointValidationError,
    HistoricalObjectiveMigration,
    JaxLearningCheckpointV3,
    inspect_historical_v3_objective_alias,
    load_learning_checkpoint_v3,
    save_learning_checkpoint_v3,
)

__all__ = [
    "CONTINUATION_CHECKPOINT_ROLE",
    "CheckpointPayloadDescriptor",
    "CHECKPOINT_OPTIMIZER_STEP_MISMATCH",
    "CHECKPOINT_OBJECTIVE_IDENTITY_MISSING",
    "CHECKPOINT_V3_SCHEMA_VERSION",
    "CheckpointValidationError",
    "HistoricalObjectiveMigration",
    "HISTORICAL_MSE_OBJECTIVE_ALIASES",
    "FutureTensorPayloadDescriptor",
    "HF_DISTRIBUTION_CHECKPOINT_ROLE",
    "LearningCheckpoint",
    "JaxLearningCheckpointV3",
    "load_learning_checkpoint",
    "reject_implicit_hf_conversion",
    "save_learning_checkpoint",
    "load_learning_checkpoint_v3",
    "inspect_historical_v3_objective_alias",
    "save_learning_checkpoint_v3",
]
