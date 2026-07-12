"""Passive replay-artifact construction; status comes only from verification."""

from __future__ import annotations

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


__all__ = ["build_replay_receipt"]
