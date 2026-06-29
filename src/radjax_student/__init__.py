"""Student-side RADJAX training scaffold."""

from radjax_student.students.registry import StudentBackendRegistry
from radjax_student.students.tiny_debug import TinyDebugStudentBackend

__all__ = ["StudentBackendRegistry", "TinyDebugStudentBackend"]
