"""Artifact inspection and loading helpers."""

from radjax_student.artifacts.loaders import inspect_teacher_tome
from radjax_student.artifacts.targets import (
    DenseTomeTargets,
    load_dense_tome_targets,
    target_batch_from_dense_tome,
)

__all__ = [
    "DenseTomeTargets",
    "inspect_teacher_tome",
    "load_dense_tome_targets",
    "target_batch_from_dense_tome",
]
