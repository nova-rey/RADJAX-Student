from radjax_student.architecture import ArchitectureRegistry
from radjax_student.architecture.testing import FakeArchitecturePlugin
from radjax_student.debug import TinyDebugStudentBackend


def test_architecture_registry_accepts_complete_non_jax_plugin() -> None:
    registry = ArchitectureRegistry()
    registry.register(FakeArchitecturePlugin())
    assert registry.list_plugins() == ("test.architecture.v1",)
    assert TinyDebugStudentBackend().architecture_id == "tiny_debug"
