"""Generic structured comparison for P3.11.9 replay evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from radjax_student.validation.p3_11_9_replay.canonical import canonical_digest
from radjax_student.validation.p3_11_9_replay.models import (
    ReplayArmEvidence,
    ReplayBlocker,
    ReplayRunEvidence,
    ReplayVerificationResult,
    StatefulReplayProof,
)


def verify_replay_proof(proof: StatefulReplayProof) -> ReplayVerificationResult:
    """Verify replay equality without interpreting architecture or optimizer state."""

    blockers: list[ReplayBlocker] = []
    for mode in ("eager", "jit"):
        replay_a = proof.modes[mode]["replay_a"]
        replay_b = proof.modes[mode]["replay_b"]
        blockers.extend(_compare_runs(mode, replay_a, replay_b))
        blockers.extend(_compare_arms(mode, replay_a.uninterrupted, replay_a.resumed))
        blockers.extend(_compare_arms(mode, replay_b.uninterrupted, replay_b.resumed))
    blockers.extend(_verify_cross_mode(proof))
    return ReplayVerificationResult(tuple(blockers))


def verify_recorded_artifact(
    generated: bytes, recorded: bytes
) -> ReplayVerificationResult:
    if generated == recorded:
        return ReplayVerificationResult()
    return ReplayVerificationResult(
        (
            ReplayBlocker(
                "replay_receipt_mismatch",
                "recorded_artifact",
                canonical_digest({"bytes": recorded.hex()}),
                canonical_digest({"bytes": generated.hex()}),
            ),
        )
    )


def recorded_artifact_difference(generated: bytes, recorded: bytes) -> str:
    """Return the first stable structural difference without exposing payloads.

    This is diagnostic evidence for a failed read-only gate.  It deliberately
    reports only a field path and canonical identities, never model arrays or
    checkpoint contents.
    """

    try:
        import json

        expected = json.loads(recorded)
        observed = json.loads(generated)
        field, expected_value, observed_value = _first_difference(expected, observed)
        # The receipt's top-level evidence digest necessarily changes after a
        # nested difference. Report that underlying difference instead.
        if field == "evidence_digest":
            expected.pop("evidence_digest", None)
            observed.pop("evidence_digest", None)
            field, expected_value, observed_value = _first_difference(
                expected, observed
            )
    except (UnicodeDecodeError, ValueError, TypeError):
        return "artifact_bytes"
    return (
        f"{field} expected={canonical_digest({'value': expected_value})} "
        f"observed={canonical_digest({'value': observed_value})}"
    )


def _compare_runs(
    mode: str, expected: ReplayRunEvidence, observed: ReplayRunEvidence
) -> list[ReplayBlocker]:
    if expected.digest == observed.digest:
        return []
    return _difference_blockers(
        expected.to_dict(), observed.to_dict(), mode=mode, arm=None, prefix="replay"
    )


def _compare_arms(
    mode: str, uninterrupted: ReplayArmEvidence, resumed: ReplayArmEvidence
) -> list[ReplayBlocker]:
    """Compare state/results while omitting the intentional arm/restore labels."""

    if not resumed.restore_used_caller_identity:
        return [
            ReplayBlocker(
                "replay_checkpoint_identity_mismatch",
                "resumed.restore_used_caller_identity",
                canonical_digest({"value": True}),
                canonical_digest({"value": False}),
                mode,
                "resumed",
            )
        ]
    expected = uninterrupted.to_dict()
    observed = resumed.to_dict()
    for payload in (expected, observed):
        payload.pop("arm")
        payload.pop("restore_used_caller_identity")
    if canonical_digest(expected) == canonical_digest(observed):
        return []
    return _difference_blockers(
        expected,
        observed,
        mode=mode,
        arm="resumed",
        prefix="same_mode_resume",
    )


def _verify_cross_mode(proof: StatefulReplayProof) -> list[ReplayBlocker]:
    evidence = proof.cross_mode
    required = {
        "structure_equal",
        "dtype_shape_equal",
        "integer_values_equal",
        "floating_values_within_tolerance",
        "lifecycle_identity_equal",
        "hook_metric_paths_equal",
        "rng_identity_equal",
        "runtime_structure_equal",
    }
    if set(evidence) != required or not all(evidence.values()):
        return [
            ReplayBlocker(
                "replay_runtime_identity_mismatch",
                "cross_mode",
                canonical_digest({"required": sorted(required)}),
                canonical_digest(dict(evidence)),
            )
        ]
    return []


def _difference_blockers(
    expected: Any,
    observed: Any,
    *,
    mode: str,
    arm: str | None,
    prefix: str,
) -> list[ReplayBlocker]:
    field, expected_value, observed_value = _first_difference(expected, observed)
    code = _code_for_field(field)
    step_index = _step_index(field)
    return [
        ReplayBlocker(
            code,
            f"{prefix}.{field}",
            canonical_digest({"value": expected_value}),
            canonical_digest({"value": observed_value}),
            mode,
            arm,
            step_index,
        )
    ]


def _first_difference(
    expected: Any, observed: Any, prefix: str = ""
) -> tuple[str, Any, Any]:
    if type(expected) is not type(observed):
        return prefix or "value", expected, observed
    if isinstance(expected, Mapping):
        if set(expected) != set(observed):
            return prefix or "fields", sorted(expected), sorted(observed)
        for key in sorted(expected):
            result = _first_difference(
                expected[key], observed[key], f"{prefix}.{key}".strip(".")
            )
            if result[0] != "":
                return result
        return "", expected, observed
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return prefix or "items", expected, observed
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            result = _first_difference(left, right, f"{prefix}[{index}]")
            if result[0] != "":
                return result
        return "", expected, observed
    if expected != observed:
        return prefix or "value", expected, observed
    return "", expected, observed


def _code_for_field(field: str) -> str:
    checks = (
        ("batch", "replay_batch_sequence_mismatch"),
        ("lifecycle", "replay_lifecycle_identity_mismatch"),
        ("rng", "replay_rng_identity_mismatch"),
        ("checkpoint", "replay_checkpoint_identity_mismatch"),
        ("parameter", "replay_parameter_state_mismatch"),
        ("architecture_carry", "replay_architecture_carry_mismatch"),
        ("optimizer", "replay_optimizer_state_mismatch"),
        ("learning", "replay_learning_state_mismatch"),
        ("scope", "replay_scope_routing_mismatch"),
        ("objective", "replay_scope_routing_mismatch"),
        ("metric", "replay_metric_mismatch"),
        ("hook", "replay_hook_sequence_mismatch"),
        ("report", "replay_report_mismatch"),
        ("runtime", "replay_runtime_identity_mismatch"),
        ("steps", "replay_step_identity_mismatch"),
    )
    for marker, code in checks:
        if marker in field:
            return code
    return "replay_experiment_identity_mismatch"


def _step_index(field: str) -> int | None:
    marker = "steps["
    if marker not in field:
        return None
    suffix = field.split(marker, 1)[1].split("]", 1)[0]
    return int(suffix) if suffix.isdigit() else None


__all__ = ["verify_recorded_artifact", "verify_replay_proof"]
