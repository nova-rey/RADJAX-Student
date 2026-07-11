from __future__ import annotations

import importlib
import warnings

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.debug import TinyDebugStudentBackend


def test_architecture_registry_is_the_public_architecture_registry():
    assert ArchitectureRegistry.__module__ == "radjax_student.architecture.registry"


def test_debug_backend_has_explicit_namespace():
    assert TinyDebugStudentBackend().architecture_id == "tiny_debug"


def test_students_namespace_is_only_a_deprecation_compatibility_path():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        import radjax_student.students as students

        students = importlib.reload(students)
        StudentBackendRegistry = students.StudentBackendRegistry

        assert StudentBackendRegistry.with_defaults().names() == ("tiny_debug",)
    assert any(item.category is DeprecationWarning for item in recorded)
