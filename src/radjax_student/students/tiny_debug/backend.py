"""Deprecated compatibility import for the old students namespace."""

import warnings

from radjax_student.debug.tiny_debug import TinyDebugStudentBackend

warnings.warn(
    "radjax_student.students.tiny_debug is deprecated; use "
    "radjax_student.debug.tiny_debug",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TinyDebugStudentBackend"]
