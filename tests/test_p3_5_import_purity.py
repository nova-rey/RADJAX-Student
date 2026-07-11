from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "radjax_student"
FORBIDDEN = {
    "jax",
    "jaxlib",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "radjax_tome",
}


def _fresh_import(module: str) -> set[str]:
    script = f"""
import importlib
import sys
importlib.import_module({module!r})
print("\\n".join(sorted({{name.split('.', 1)[0] for name in sys.modules}})))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.splitlines())


def test_base_package_imports_are_pure_in_fresh_subprocesses():
    for module in (
        "radjax_student",
        "radjax_student.architecture",
        "radjax_student.learning",
    ):
        assert _fresh_import(module).isdisjoint(FORBIDDEN), module


def test_jax_isolated_to_explicit_adapter_source():
    for path in SOURCE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.relative_to(SOURCE_ROOT).as_posix() in {
            "learning/jax_core.py",
            "learning/p3_5_acceptance.py",
        }:
            continue
        assert "import jax" not in source
        assert "from jax" not in source


def test_production_roots_have_no_legacy_reexports():
    artifact_init = (SOURCE_ROOT / "artifacts" / "__init__.py").read_text(
        encoding="utf-8"
    )
    training_init = (SOURCE_ROOT / "training" / "__init__.py").read_text(
        encoding="utf-8"
    )
    package_init = (SOURCE_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "load_dense_tome_targets" not in artifact_init
    assert "run_tiny_train_step" not in training_init
    assert "students" not in package_init


def test_runtime_and_architecture_do_not_cross_implementation_boundaries():
    for package in ("architecture", "runtime", "learning"):
        for path in (SOURCE_ROOT / package).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports = {
                node.module or alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            }
            imports.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            assert not any(
                name.startswith("radjax_student.students") for name in imports
            )
            assert not any(name.startswith("radjax_tome") for name in imports)
    jax_source = (SOURCE_ROOT / "learning" / "jax_core.py").read_text(encoding="utf-8")
    jax_tree = ast.parse(jax_source)
    jax_imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(jax_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    jax_imports.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(jax_tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "numpy" not in jax_imports
    assert "jax.jit" not in jax_source
