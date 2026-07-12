"""Explicit compatibility and historical smoke implementations."""

from radjax_student.legacy.scalar_learning import (
    LegacyScalarObjective,
    LegacyScalarObjectiveAdapter,
    LegacyScalarStepExecution,
    legacy_scalar_learning_step,
    run_legacy_learning_loop,
)

__all__ = [
    "LegacyScalarObjective",
    "LegacyScalarObjectiveAdapter",
    "LegacyScalarStepExecution",
    "legacy_scalar_learning_step",
    "run_legacy_learning_loop",
]
