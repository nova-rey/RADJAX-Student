"""Deprecated compatibility import for the former training smoke."""

from __future__ import annotations

import warnings

from radjax_student.legacy.training import TinyTrainStepResult, run_tiny_train_step

warnings.warn(
    "radjax_student.training.distill is deprecated; use radjax_student.legacy.training",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TinyTrainStepResult", "run_tiny_train_step"]
