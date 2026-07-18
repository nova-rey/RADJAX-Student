"""Focused JAX-free foundation-closure policy tests."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from radjax_student.validation.foundation_audit_closure import (
    CANONICAL_TRAINING_PATHS,
    SCHEMA_VERSION,
    audit_source_fixture,
    build_foundation_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def test_foundation_audit_is_clean_and_uses_literal_canonical_paths() -> None:
    report = build_foundation_audit(ROOT)
    assert report.status == "pass"
    assert report.to_dict()["schema_version"] == SCHEMA_VERSION
    assert "learning/composition.py" in CANONICAL_TRAINING_PATHS
    assert "steps/jax_step.py" in CANONICAL_TRAINING_PATHS


def test_literal_source_fixtures_reject_forbidden_foundation_edges() -> None:
    assert audit_source_fixture(
        "from radjax_student.steps import jax_step\n", relative_path="runtime/x.py"
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from radjax_student.validation import compatibility\n",
        relative_path="reports/x.py",
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "import numpy\n", relative_path="steps/jax_step.py"
    ) == ("canonical_jax_purity",)
    assert audit_source_fixture(
        "from radjax_student.legacy.losses import dense_kl_loss\n",
        relative_path="learning/assembly.py",
    ) == ("canonical_numpy_loss_import",)
    assert audit_source_fixture(
        "from radjax_student.validation import gate\n",
        relative_path="cli/inspect.py",
    ) == ("production_validation_import",)


def test_runtime_import_and_local_test_support_are_hermetic() -> None:
    script = "\n".join(
        (
            "import importlib.util",
            "import pathlib",
            "import sys",
            "import radjax_student.runtime.callables",
            "assert not any("
            "name.startswith('radjax_student.steps') for name in sys.modules"
            ")",
            "spec = importlib.util.find_spec('tests.support.linear_objective')",
            "assert spec is not None and spec.origin is not None",
            "assert pathlib.Path(spec.origin).resolve().is_relative_to("
            "pathlib.Path.cwd() / 'tests'"
            ")",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert importlib.util.find_spec("radjax_student.losses") is None
