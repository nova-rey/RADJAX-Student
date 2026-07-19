import tomllib
from pathlib import Path


def test_package_imports() -> None:
    import radjax_student

    assert radjax_student.__all__ == []
    assert not hasattr(radjax_student, "TinyDebugStudentBackend")


def test_runtime_and_test_jax_constraints_are_aligned() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    runtime_jax = extras["jax"]
    test_jax = [
        requirement
        for requirement in extras["test-jax"]
        if requirement.startswith("jax[cpu]")
    ]

    assert runtime_jax == ["jax[cpu]==0.4.38"]
    assert test_jax == runtime_jax
