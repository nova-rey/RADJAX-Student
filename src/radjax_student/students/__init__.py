"""Deprecated compatibility namespace for pre-P3.5 student backends."""

import warnings

from radjax_student.students.base import StudentBackend
from radjax_student.students.registry import StudentBackendRegistry

warnings.warn(
    "radjax_student.students is deprecated; use radjax_student.architecture or "
    "radjax_student.debug",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["StudentBackend", "StudentBackendRegistry"]
