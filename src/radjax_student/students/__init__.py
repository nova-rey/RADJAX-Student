"""Student backend protocols and registry."""

from radjax_student.students.base import StudentBackend
from radjax_student.students.registry import StudentBackendRegistry

__all__ = ["StudentBackend", "StudentBackendRegistry"]
