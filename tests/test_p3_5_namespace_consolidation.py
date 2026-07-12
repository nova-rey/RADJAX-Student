from __future__ import annotations

import importlib

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.debug import TinyDebugStudentBackend


def test_architecture_registry_is_the_public_architecture_registry():
    assert ArchitectureRegistry.__module__ == "radjax_student.architecture.registry"


def test_debug_backend_has_explicit_namespace():
    assert TinyDebugStudentBackend().architecture_id == "tiny_debug"


def test_students_namespace_has_been_removed():
    assert importlib.util.find_spec("radjax_student.students") is None
