"""Passive replay-artifact construction; status comes only from verification."""

from __future__ import annotations

from radjax_student.validation.p3_11_9_replay.canonical import ReplayCanonicalError
from radjax_student.validation.p3_11_9_replay.models import (
    StatefulReplayProof,
    StatefulReplayReceipt,
)
from radjax_student.validation.p3_11_9_replay.verifier import verify_replay_proof


def build_replay_receipt(proof: StatefulReplayProof) -> StatefulReplayReceipt:
    verification = verify_replay_proof(proof)
    if not verification.passed:
        details = ", ".join(item.code for item in verification.blockers)
        raise ValueError(f"replay verification failed: {details}")
    return StatefulReplayReceipt(proof, verification)


def validate_recorded_replay_artifact(data: bytes | str) -> dict[str, object]:
    """Parse a recorded artifact and reject unearned comparison claims.

    This remains a passive validation boundary. It does not execute JAX, but it
    makes every recorded cross-mode claim and embedded verifier state explicit
    before an acceptance consumer can use the artifact.
    """

    payload = StatefulReplayReceipt.from_json_bytes(data)
    for mode in ("eager", "jit"):
        mode_payload = payload["modes"][mode]
        if mode_payload["replay_a_digest"] != mode_payload["replay_b_digest"]:
            raise ReplayCanonicalError(
                "recorded replay A/B identities are not exactly equal"
            )
        if (
            mode_payload["uninterrupted_arm_digest"]
            == mode_payload["resumed_arm_digest"]
        ):
            raise ReplayCanonicalError(
                "recorded resumed arm lacks an independent resume identity"
            )
    cross_mode = payload["cross_mode"]
    if not isinstance(cross_mode, dict) or not all(
        value is True
        for name, value in cross_mode.items()
        if name
        not in {
            "declared_rtol",
            "declared_atol",
        }
    ):
        raise ReplayCanonicalError("recorded cross-mode evidence is not passing")
    verifier = payload["verifier"]
    if (
        not isinstance(verifier, dict)
        or verifier.get("status") != "pass"
        or verifier.get("blockers") != []
    ):
        raise ReplayCanonicalError("recorded replay verifier evidence is not passing")
    return dict(payload)


__all__ = ["build_replay_receipt", "validate_recorded_replay_artifact"]
