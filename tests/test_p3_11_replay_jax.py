"""Executed-conveyor and cross-mode tests for P3.11.9 replay evidence."""

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
jnp = pytest.importorskip("jax.numpy")

from radjax_student.validation.p3_11_9_replay.runner_jax import (  # noqa: E402
    _compare_executed_cross_mode,
    execute_replay,
    execute_stateful_replays,
)
from radjax_student.validation.p3_11_9_replay.verifier import (  # noqa: E402
    verify_replay_proof,
)

pytestmark = pytest.mark.jax

ROOT = Path(__file__).resolve().parents[1]


def _changed_digest() -> str:
    return "0" * 64


def _cross_mode_captures(tmp_path):
    _, eager, _, _ = execute_replay("eager", tmp_path / "eager")
    _, jit, _, _ = execute_replay("jit", tmp_path / "jit")
    proof = execute_stateful_replays(tmp_path / "proof")
    return proof, eager, jit


def _comparison(proof, eager, jit):
    return _compare_executed_cross_mode(eager, jit, proof.tolerance)


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


def test_cross_mode_result_is_computed_from_executed_arrays_and_metrics(tmp_path):
    proof, eager, jit = _cross_mode_captures(tmp_path)
    comparison = _comparison(proof, eager, jit)
    assert comparison == proof.cross_mode
    assert all(getattr(comparison, field) for field in comparison._BOOLEAN_FIELDS)
    assert comparison.declared_rtol == (1e-6).hex()
    assert comparison.declared_atol == (1e-6).hex()


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (
            lambda eager: replace(
                eager,
                parameters={
                    **eager.parameters,
                    "trunk": {"weight": eager.parameters["trunk"]["weight"] + 1.0},
                },
            ),
            "floating_values_within_tolerance",
        ),
        (
            lambda eager: replace(
                eager,
                optimizer_arrays={
                    **eager.optimizer_arrays,
                    "step": eager.optimizer_arrays["step"] + 1,
                },
            ),
            "integer_values_equal",
        ),
        (
            lambda eager: replace(
                eager,
                architecture_carry={
                    **eager.architecture_carry,
                    "rng_probe": eager.architecture_carry["rng_probe"].astype(
                        jnp.int32
                    ),
                },
            ),
            "dtype_shape_equal",
        ),
        (
            lambda eager: replace(
                eager,
                parameters={
                    **eager.parameters,
                    "trunk": {
                        "weight": jnp.reshape(
                            eager.parameters["trunk"]["weight"], (1, 1)
                        )
                    },
                },
            ),
            "dtype_shape_equal",
        ),
        (
            lambda eager: replace(
                eager,
                parameters={"trunk": eager.parameters["trunk"]},
            ),
            "structure_equal",
        ),
        (
            lambda eager: replace(
                eager,
                retained_metrics=(
                    (eager.retained_metrics[0][0], eager.retained_metrics[0][1] + 1.0),
                    *eager.retained_metrics[1:],
                ),
            ),
            "metric_values_within_tolerance",
        ),
        (
            lambda eager: replace(
                eager,
                retained_metrics=eager.retained_metrics[1:],
            ),
            "metric_names_equal",
        ),
        (
            lambda eager: replace(
                eager,
                hook_events=tuple(reversed(eager.hook_events)),
            ),
            "hook_sequence_equal",
        ),
        (
            lambda eager: replace(
                eager,
                logical_paths=(
                    (("foreign.path",), eager.logical_paths[0][1]),
                    *eager.logical_paths[1:],
                ),
            ),
            "logical_paths_equal",
        ),
        (
            lambda eager: replace(
                eager,
                rng_coordinates=(
                    replace(
                        eager.rng_coordinates[0],
                        invocation_index=eager.rng_coordinates[0].invocation_index + 1,
                    ),
                    *eager.rng_coordinates[1:],
                ),
            ),
            "rng_identity_equal",
        ),
        (
            lambda eager: replace(
                eager,
                runtimes=(
                    replace(eager.runtimes[0], precision_policy="bfloat16"),
                    *eager.runtimes[1:],
                ),
            ),
            "runtime_structure_equal",
        ),
        (
            lambda eager: replace(
                eager,
                runtimes=(
                    replace(
                        eager.runtimes[0],
                        output_metadata_fields=eager.runtimes[0].output_metadata_fields[
                            1:
                        ],
                    ),
                    *eager.runtimes[1:],
                ),
            ),
            "runtime_structure_equal",
        ),
        (
            lambda eager: replace(
                eager,
                evidence=replace(
                    eager.evidence,
                    lifecycle_identity=replace(
                        eager.evidence.lifecycle_identity,
                        parameter_catalog_digest=_changed_digest(),
                        hf_reference=replace(
                            eager.evidence.lifecycle_identity.hf_reference,
                            parameter_catalog_digest=_changed_digest(),
                        ),
                    ),
                ),
            ),
            "lifecycle_identity_equal",
        ),
    ],
)
def test_cross_mode_comparison_detects_each_executed_drift(tmp_path, mutation, field):
    proof, eager, jit = _cross_mode_captures(tmp_path)
    assert getattr(_comparison(proof, mutation(eager), jit), field) is False


def test_verifier_rejects_typed_cross_mode_drift_and_hardcoded_claim(tmp_path):
    proof = execute_stateful_replays(tmp_path)
    failed_comparison = replace(
        proof.cross_mode, floating_values_within_tolerance=False
    )
    result = verify_replay_proof(
        replace(proof, cross_mode=failed_comparison, executed_cross_mode=None)
    )
    assert result.blockers[0].code == "replay_cross_mode_float_mismatch"
    hardcoded = replace(proof.cross_mode, floating_values_within_tolerance=True)
    executed = replace(proof.cross_mode, floating_values_within_tolerance=False)
    result = verify_replay_proof(
        replace(proof, cross_mode=hardcoded, executed_cross_mode=executed)
    )
    assert result.blockers[0].code == "replay_cross_mode_runtime_mismatch"
    with pytest.raises(ValueError, match="replay verification failed"):
        build_replay_receipt(
            replace(proof, cross_mode=hardcoded, executed_cross_mode=executed)
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
