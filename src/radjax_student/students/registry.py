from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from radjax_student.students.base import StudentBackend
from radjax_student.students.tiny_debug.backend import TinyDebugStudentBackend

warnings.warn(
    "StudentBackendRegistry is deprecated; use ArchitectureRegistry for "
    "architecture plugins",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class StudentBackendRegistry:
    _backends: dict[str, StudentBackend] = field(default_factory=dict)

    @classmethod
    def with_defaults(cls) -> StudentBackendRegistry:
        registry = cls()
        registry.register(TinyDebugStudentBackend())
        return registry

    def register(self, backend: StudentBackend) -> None:
        self._backends[backend.architecture_id] = backend

    def get(self, architecture_id: str) -> StudentBackend:
        try:
            return self._backends[architecture_id]
        except KeyError as exc:
            raise KeyError(f"unknown student backend: {architecture_id}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))
