from __future__ import annotations

import os
import socket
import subprocess
import sys
import urllib.request

from tome_fixtures import write_dense_tome

from radjax_student.artifacts import open_tome_artifact
from radjax_student.artifacts import targets as target_loading
from radjax_student.legacy import training as legacy_training
from radjax_student.students.registry import StudentBackendRegistry
from radjax_student.students.tiny_debug import TinyDebugStudentBackend
from radjax_student.validation import infer_run_defaults_from_tome

from .support import REPO_ROOT, canonical_fixture, run_cli

FORBIDDEN_DEFAULT_IMPORTS = (
    "jax",
    "radjax_tome",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
)


def test_default_source_has_no_optional_or_producer_imports() -> None:
    source_root = REPO_ROOT / "src" / "radjax_student"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        if path.relative_to(source_root).as_posix() in {
            "learning/jax_core.py",
            "learning/p3_5_acceptance.py",
        }:
            continue
        source = path.read_text(encoding="utf-8")
        for dependency in FORBIDDEN_DEFAULT_IMPORTS:
            if f"import {dependency}" in source or f"from {dependency}" in source:
                offenders.append(f"{path.relative_to(source_root)}: {dependency}")

    assert offenders == []


def test_phase1_pipeline_imports_without_optional_ml_dependencies() -> None:
    script = """
import builtins
real_import = builtins.__import__
forbidden = {"jax", "radjax_tome", "torch", "transformers", "datasets", "accelerate"}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from radjax_contract.testing import production_tome_fixture_path
from radjax_student.cli.main import main
assert main(("doctor",)) == 0
assert main(("inspect", "--tome", str(production_tome_fixture_path()))) == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_phase1_commands_do_not_consume_payload_or_execute_runtime(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("execution-only API was called")

    monkeypatch.setattr(target_loading, "load_dense_tome_targets", forbidden)
    monkeypatch.setattr(TinyDebugStudentBackend, "__init__", forbidden)
    monkeypatch.setattr(TinyDebugStudentBackend, "init", forbidden)
    monkeypatch.setattr(StudentBackendRegistry, "with_defaults", forbidden)
    monkeypatch.setattr(legacy_training, "run_tiny_train_step", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    sys.modules.pop("radjax_student.runtime", None)
    sys.modules.pop("radjax_student.schedules", None)

    inspect_code, _, inspect_error = run_cli(
        "inspect",
        "--tome",
        str(canonical_fixture()),
    )
    doctor_code, _, doctor_error = run_cli("doctor")

    assert inspect_code == 1
    assert inspect_error == ""
    assert doctor_code == 0
    assert doctor_error == ""
    assert "radjax_student.runtime" not in sys.modules
    assert "radjax_student.schedules" not in sys.modules


def test_production_view_exposes_metadata_not_contract_payload_objects() -> None:
    view = open_tome_artifact(canonical_fixture())
    production_student_sources = (
        REPO_ROOT / "src/radjax_student/artifacts/view.py",
        REPO_ROOT / "src/radjax_student/reports/inspection.py",
    )

    assert view.corridor_contract is not None
    assert view.exemplar_contract is not None
    assert not hasattr(view.corridor_contract, "assignments")
    assert not hasattr(view.corridor_contract, "fingerprints")
    assert not hasattr(view.exemplar_contract, "payloads")
    assert not hasattr(view, "model")
    assert not hasattr(view, "runtime")
    assert not hasattr(view, "schedule")
    assert not hasattr(view, "checkpoint")
    for source_path in production_student_sources:
        source = source_path.read_text(encoding="utf-8")
        assert "np.load" not in source
        assert ".payloads" not in source
        assert "read_json" not in source


def test_legacy_dense_v0_remains_isolated_smoke_only(tmp_path) -> None:
    legacy = write_dense_tome(tmp_path / "legacy")
    legacy_defaults = infer_run_defaults_from_tome(legacy)
    production_defaults = infer_run_defaults_from_tome(canonical_fixture())

    assert legacy_defaults.artifact_facts.contract_family == "legacy_dense_v0"
    assert legacy_defaults.legacy_smoke_defaults is not None
    assert legacy_defaults.legacy_smoke_defaults["classification"] == (
        "legacy_dense_v0_smoke_only"
    )
    assert production_defaults.artifact_facts.contract_family == "production_v2"
    assert production_defaults.legacy_smoke_defaults is None
    assert production_defaults.available_surfaces
    assert "payload_format" not in production_defaults.artifact_facts.to_dict()
    assert "adapter" not in production_defaults.artifact_facts.to_dict()
    assert "Deprecated tiny smoke shim" in (
        REPO_ROOT / "src/radjax_student/cli/train_student.py"
    ).read_text(encoding="utf-8")
