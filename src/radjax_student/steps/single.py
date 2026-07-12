"""Deprecated one-way compatibility shim for the scalar learning step."""

from __future__ import annotations

import warnings

from radjax_student.legacy.scalar_learning import (
    LegacyScalarObjective as ScalarObjective,
)
from radjax_student.legacy.scalar_learning import (
    LegacyScalarObjectiveAdapter,
    legacy_scalar_learning_step,
)
from radjax_student.legacy.scalar_learning import (
    LegacyScalarStepExecution as SingleStepExecution,
)


def learning_step(*args, **kwargs):
    """Deprecated alias for the explicit legacy scalar step."""

    warnings.warn(
        "radjax_student.steps.single.learning_step is deprecated; use "
        "radjax_student.legacy.scalar_learning.legacy_scalar_learning_step",
        DeprecationWarning,
        stacklevel=2,
    )
    return legacy_scalar_learning_step(*args, **kwargs)


__all__ = [
    "LegacyScalarObjectiveAdapter",
    "ScalarObjective",
    "SingleStepExecution",
    "learning_step",
]
