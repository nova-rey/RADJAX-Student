"""Focused JAX-free foundation-closure policy tests."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.validation.foundation_audit_closure import (
    CANONICAL_TRAINING_PATHS,
    SCHEMA_VERSION,
    _bytes,
    _p312b_recorded_evidence_current,
    audit_hf_authority_fixture,
    audit_source_fixture,
    build_foundation_audit,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.jax
def test_foundation_audit_is_clean_and_uses_literal_canonical_paths() -> None:
    report = build_foundation_audit(ROOT)
    assert report.status == "pass"
    assert report.to_dict()["schema_version"] == SCHEMA_VERSION
    assert "recorded_gates_read_only" not in report.to_dict()
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
    assert audit_source_fixture(
        "def execute_p3_99_proof(): pass\n",
        relative_path="learning/new_acceptance.py",
    ) == ("new_production_proof_module:learning/new_acceptance.py",)
    assert (
        audit_source_fixture(
            "def execute_p3_5_proof(): pass\n",
            relative_path="learning/p3_5_acceptance.py",
        )
        == ()
    )
    assert audit_source_fixture(
        "from radjax_student.learning import assembly\n",
        relative_path="runtime/x.py",
    ) == ("runtime_learning_import",)


def test_hf_authority_ast_rejects_independent_path_breakages() -> None:
    source = ROOT / "src" / "radjax_student"
    paths = (
        "architecture/models.py",
        "learning/assembly.py",
        "steps/jax_loop.py",
        "checkpoints/v3.py",
        "validation/p3_11_9_replay/runner_jax.py",
        "learning/run_report.py",
    )
    sources = {path: (source / path).read_text(encoding="utf-8") for path in paths}
    assert audit_hf_authority_fixture(sources) == ()

    assembly = dict(sources)
    assembly["learning/assembly.py"] = assembly["learning/assembly.py"].replace(
        "hf_descriptor=initialized.hf_descriptor", "hf_descriptor=foreign_descriptor"
    )
    assert audit_hf_authority_fixture(assembly) == (
        "hf_assembly_descriptor_substitution",
    )

    checkpoint = dict(sources)
    checkpoint["checkpoints/v3.py"] = checkpoint["checkpoints/v3.py"].replace(
        "if expected_hf_descriptor is None:", "if False:"
    )
    assert audit_hf_authority_fixture(checkpoint) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    comparison = dict(sources)
    comparison["checkpoints/v3.py"] = comparison["checkpoints/v3.py"].replace(
        "if hf_descriptor != expected_hf_descriptor:", "if False:"
    )
    assert audit_hf_authority_fixture(comparison) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    lifecycle = dict(sources)
    lifecycle["steps/jax_loop.py"] = lifecycle["steps/jax_loop.py"].replace(
        "expected_hf_descriptor=self.hf_descriptor",
        "expected_hf_descriptor=None",
    )
    assert audit_hf_authority_fixture(lifecycle) == ("hf_checkpoint_lifecycle_bypass",)

    replay = dict(sources)
    replay["validation/p3_11_9_replay/runner_jax.py"] = replay[
        "validation/p3_11_9_replay/runner_jax.py"
    ].replace("lifecycle.hf_descriptor", "foreign_descriptor")
    assert audit_hf_authority_fixture(replay) == ("hf_replay_non_lifecycle_descriptor",)

    report = dict(sources)
    report["learning/run_report.py"] = report["learning/run_report.py"].replace(
        "validate_hf_descriptor_match", "compare_unchecked_descriptor"
    )
    assert audit_hf_authority_fixture(report) == (
        "hf_report_descriptor_validation_bypassed",
    )

    report_operands = dict(sources)
    report_operands["learning/run_report.py"] = report_operands[
        "learning/run_report.py"
    ].replace(
        "validate_hf_descriptor_match(executed_descriptor, summary.descriptor)",
        "validate_hf_descriptor_match(foreign_descriptor, foreign_summary)",
    )
    assert audit_hf_authority_fixture(report_operands) == (
        "hf_report_descriptor_validation_bypassed",
    )


@pytest.mark.jax
def test_foundation_report_bytes_are_deterministic_and_public_checker_detects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    report = build_foundation_audit(ROOT)
    generated = _bytes(report)
    assert generated == _bytes(build_foundation_audit(ROOT))
    recorded = json.loads(
        (ROOT / "docs/FOUNDATION_AUDIT_CLOSURE_REPORT.json").read_text()
    )
    assert recorded["schema_version"] == SCHEMA_VERSION
    corrupted = tmp_path / "FOUNDATION_AUDIT_CLOSURE_REPORT.json"
    corrupted.write_bytes(generated[:-1] + b" ")
    assert main(["--check-recorded", "--recorded", str(corrupted)]) == 1


@pytest.mark.jax
def test_foundation_rejects_schema_valid_stale_p312b_receipt(tmp_path: Path) -> None:
    receipt = json.loads(
        (ROOT / "docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
    )
    receipt["descriptor_digest"] = "0" * 64
    receipt["checkpoint_hf_descriptor_digest"] = "0" * 64
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    assert not _p312b_recorded_evidence_current(tmp_path)


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
