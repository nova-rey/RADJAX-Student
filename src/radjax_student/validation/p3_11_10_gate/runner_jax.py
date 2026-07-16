"""Lazy JAX execution bridge for P3.11.10.

The gate deliberately reuses the accepted P3.11.9 public runner.  It does not
implement another learning path or duplicate model mathematics.
"""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

_EVIDENCE_CACHE: dict[str, dict[str, Any]] = {}


def execute_positive(repository_root: Path) -> dict[str, Any]:
    cache_key = str(repository_root.resolve())
    cached = _EVIDENCE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    del repository_root
    import tempfile

    from radjax_student.validation.p3_11_9_replay.runner_jax import (
        execute_stateful_replays,
    )

    with tempfile.TemporaryDirectory(prefix="radjax-p31110-jax-") as temporary:
        proof = execute_stateful_replays(Path(temporary))
    from radjax_student.validation.p3_11_9_replay.artifact import (
        build_replay_receipt,
    )

    receipt = build_replay_receipt(proof).to_dict()
    evidence = {
        "replay_evidence_digest": receipt["evidence_digest"],
        "modes": tuple(sorted(proof.modes)),
        "cross_mode_digest": receipt["cross_mode"],
    }
    _EVIDENCE_CACHE[cache_key] = dict(evidence)
    return evidence


__all__ = ["execute_positive"]
