"""Real-conveyor deterministic replay tests for P3.11.9."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from radjax_student.validation.p3_11_9_replay.artifact import build_replay_receipt

jax = pytest.importorskip("jax")

from radjax_student.validation.p3_11_9_replay.runner_jax import (  # noqa: E402
    execute_stateful_replays,
)
from radjax_student.validation.p3_11_9_replay.verifier import (  # noqa: E402
    verify_replay_proof,
)

pytestmark = pytest.mark.jax

ROOT = Path(__file__).resolve().parents[1]


def _changed_digest() -> str:
    return "0" * 64


def _replace_arm(proof, *, mode="eager", replay="replay_a", **changes):
    run = proof.modes[mode][replay]
    arm = replace(run.uninterrupted, **changes)
    modes = {key: dict(value) for key, value in proof.modes.items()}
    modes[mode][replay] = replace(run, uninterrupted=arm)
    return replace(proof, modes=modes)


def test_real_replays_are_exact_and_artifact_generation_is_byte_identical(tmp_path):
    first = execute_stateful_replays(tmp_path / "first")
    second = execute_stateful_replays(tmp_path / "second")
    first_artifact = build_replay_receipt(first).to_json_bytes()
    second_artifact = build_replay_receipt(second).to_json_bytes()
    assert first_artifact == second_artifact
    assert verify_replay_proof(first).passed
    for mode in ("eager", "jit"):
        run = first.modes[mode]["replay_a"]
        assert (
            run.uninterrupted.final_parameter_digest
            == run.resumed.final_parameter_digest
        )
        assert (
            run.uninterrupted.final_architecture_carry_digest
            == run.resumed.final_architecture_carry_digest
        )
        assert (
            run.uninterrupted.final_optimizer_array_digest
            == run.resumed.final_optimizer_array_digest
        )
        assert run.resumed.restore_used_caller_identity


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"batch_sequence_digest": _changed_digest()},
            "replay_batch_sequence_mismatch",
        ),
        ({"checkpoint_boundary": 2}, "replay_checkpoint_identity_mismatch"),
        (
            {"checkpoint_manifest_digest": _changed_digest()},
            "replay_checkpoint_identity_mismatch",
        ),
        (
            {"final_parameter_digest": _changed_digest()},
            "replay_parameter_state_mismatch",
        ),
        (
            {"final_architecture_carry_digest": _changed_digest()},
            "replay_architecture_carry_mismatch",
        ),
        (
            {"final_optimizer_array_digest": _changed_digest()},
            "replay_optimizer_state_mismatch",
        ),
        (
            {"final_learning_state_digest": _changed_digest()},
            "replay_learning_state_mismatch",
        ),
        ({"final_hook_digest": _changed_digest()}, "replay_hook_sequence_mismatch"),
        ({"retained_metrics_digest": _changed_digest()}, "replay_metric_mismatch"),
        ({"final_report_digest": _changed_digest()}, "replay_report_mismatch"),
    ],
)
def test_generic_verifier_rejects_real_structured_drift(tmp_path, changes, expected):
    proof = execute_stateful_replays(tmp_path)
    result = verify_replay_proof(_replace_arm(proof, **changes))
    assert not result.passed
    assert result.blockers[0].code == expected


def test_verifier_rejects_rng_scope_runtime_and_restore_drift(tmp_path):
    proof = execute_stateful_replays(tmp_path)
    run = proof.modes["eager"]["replay_a"]
    first_step = run.uninterrupted.steps[0]
    changed_rng = replace(first_step, rng={**first_step.rng, "slot": "augmentation"})
    changed_scope = replace(first_step, update_scope_digest=_changed_digest())
    changed_runtime = replace(
        first_step, runtime={**first_step.runtime, "precision_policy": "bfloat16"}
    )
    for step, expected in (
        (changed_rng, "replay_rng_identity_mismatch"),
        (changed_scope, "replay_scope_routing_mismatch"),
        (changed_runtime, "replay_runtime_identity_mismatch"),
    ):
        arm = replace(run.uninterrupted, steps=(step, *run.uninterrupted.steps[1:]))
        modes = {key: dict(value) for key, value in proof.modes.items()}
        modes["eager"]["replay_a"] = replace(run, uninterrupted=arm)
        result = verify_replay_proof(replace(proof, modes=modes))
        assert result.blockers[0].code == expected
    modes = {key: dict(value) for key, value in proof.modes.items()}
    for label in ("replay_a", "replay_b"):
        original = modes["eager"][label]
        modes["eager"][label] = replace(
            original,
            resumed=replace(original.resumed, restore_used_caller_identity=False),
        )
    assert (
        verify_replay_proof(replace(proof, modes=modes)).blockers[0].code
        == "replay_checkpoint_identity_mismatch"
    )


def test_check_recorded_is_repository_read_only(tmp_path):
    tracked = [
        ROOT / "docs/P3_11_9_REPLAY_EVIDENCE.json",
        ROOT / "docs/P3_11_9_DETERMINISTIC_REPLAY.md",
        ROOT / "docs/P3_5_DEPENDENCY_AUDIT.json",
        ROOT / "docs/P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json",
        ROOT / "src",
    ]
    before = {str(path): _tree_hash(path) for path in tracked}
    environment = dict(os.environ)
    # Disable interpreter cache writes so the subprocess is wholly read-only.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "radjax_student.validation.p3_11_9_replay",
            "--check-recorded",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert before == {str(path): _tree_hash(path) for path in tracked}


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()
