"""Deterministic P4.8 end-to-end acceptance evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from radjax_student.validation.p4_8_architecture_ingestion import (
    SCHEMA_VERSION,
    canonical_report_bytes,
    generate_phase4_report,
    write_phase4_report,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.jax


def test_phase4_report_is_byte_identical_from_fresh_execution_directories(
    tmp_path: Path,
) -> None:
    first = write_phase4_report(ROOT, tmp_path / "first-work", tmp_path / "first.json")
    second = write_phase4_report(
        ROOT, tmp_path / "second-work", tmp_path / "second.json"
    )

    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "pass"
    assert report["equation_parity_claim"] == "pinned_numpy_inference_reference"
    assert report["initialization_parity_claim"] == "not_claimed"
    assert report["training_recipe_parity_claim"] == "not_claimed"
    assert report["weight_file_compatibility"] is False
    assert all(value == "not_claimed" for value in report["non_claims"].values())
    for mode in ("eager", "jit"):
        assert report["lifecycle"][mode]["forward_finite"] is True
        assert report["lifecycle"][mode]["loss_finite"] is True
        assert report["lifecycle"][mode]["gradient_finite"] is True


def test_phase4_report_rejects_tampered_executed_identity_or_status_fact(
    tmp_path: Path,
) -> None:
    report = generate_phase4_report(ROOT, tmp_path / "work")
    mutations = (
        ("runtime_callable", "callable_id", "forged.callable"),
        ("runtime_callable", "identity_digest", "0" * 64),
        ("plugin", "architecture_id", "forged.architecture"),
        ("plugin", "architecture_version", 2),
        ("lifecycle", "jit", "compiled", False),
        ("lifecycle", "eager", "optimizer_advanced", False),
    )
    for path in mutations:
        forged = deepcopy(report)
        target = forged
        *parents, key, value = path
        for parent in parents:
            target = target[parent]
        target[key] = value
        without_digest = dict(forged)
        without_digest.pop("evidence_digest")
        forged["evidence_digest"] = sha256(
            (
                json.dumps(without_digest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest()
        with pytest.raises(ValueError):
            canonical_report_bytes(forged)


def test_p4_report_package_import_remains_jax_free() -> None:
    script = """
import builtins
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] == 'jax':
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import radjax_student.validation.p4_8_architecture_ingestion
assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
