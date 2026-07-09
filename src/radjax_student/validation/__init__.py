"""Student-side validation boundary.

This package is reserved for compatibility and readiness checks that decide
whether this runtime can consume a valid Contract artifact.
"""

from radjax_student.validation.run_defaults import (
    StudentRunDefaults,
    infer_run_defaults,
    infer_run_defaults_from_tome,
)

__all__ = [
    "StudentRunDefaults",
    "infer_run_defaults",
    "infer_run_defaults_from_tome",
]
