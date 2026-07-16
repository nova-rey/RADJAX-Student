"""Real-conveyor execution tests for the P3.11.10 gate."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")

from radjax_student.validation.p3_11_10_gate.gate import (  # noqa: E402
    build_receipt,
    execute_gate,
)
from radjax_student.validation.p3_11_10_gate.inventory import CASES  # noqa: E402

pytestmark = pytest.mark.jax
ROOT = Path(__file__).resolve().parents[1]


def test_full_gate_executes_the_complete_inventory_from_real_public_conveyor():
    proof = execute_gate(ROOT)
    receipt = build_receipt(proof).to_dict()
    cases = [case for section in proof.sections for case in section.cases]
    assert [case.definition.case_id for case in cases] == [
        case.case_id for case in CASES
    ]
    assert receipt["status"] == "pass"
    assert receipt["unexpected_pass_count"] == 0
    assert receipt["unexpected_failure_count"] == 0
    assert any(case.execution_class == "jax_executed_boundary" for case in cases)
    assert any(
        case.execution_class == "checkpoint_filesystem_adversary" for case in cases
    )
    assert any(case.execution_class == "replay_evidence_adversary" for case in cases)


def test_recorded_gate_is_read_only_and_reproduces_the_receipt():
    maintained = (
        ROOT / "docs/P3_11_10_FINAL_ADVERSARIAL_GATE_RECEIPT.json",
        ROOT / "docs/P3_11_9_REPLAY_EVIDENCE.json",
        ROOT / "docs/P3_11_INTEGRATION_CLOSURE.md",
    )
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in maintained
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "radjax_student.validation.p3_11_10_gate",
            "--check-recorded",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in maintained}
    assert after == before
