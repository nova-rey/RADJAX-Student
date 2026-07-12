"""Compatibility re-exports for neutral learning errors."""

from radjax_student.contracts.errors import (
    LEARNING_ERROR_CODES,
    LearningContractError,
    LearningErrorCode,
    LearningIssue,
)

__all__ = [
    "LEARNING_ERROR_CODES",
    "LearningContractError",
    "LearningErrorCode",
    "LearningIssue",
]
