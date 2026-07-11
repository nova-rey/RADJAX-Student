"""Explicit boundaries between continuation and future HF distribution data."""

from __future__ import annotations

from typing import NoReturn

from radjax_student.checkpoints.learning import (
    CONTINUATION_CHECKPOINT_ROLE,
    HF_DISTRIBUTION_CHECKPOINT_ROLE,
    LearningCheckpoint,
)


def reject_implicit_hf_conversion(checkpoint: LearningCheckpoint) -> NoReturn:
    """Refuse to use a continuation checkpoint as an HF distribution file."""

    if checkpoint.role != CONTINUATION_CHECKPOINT_ROLE:
        raise ValueError("checkpoint is not a RADJAX continuation checkpoint")
    raise ValueError(
        "explicit HF distribution conversion is not implemented; continuation "
        f"role cannot be treated as {HF_DISTRIBUTION_CHECKPOINT_ROLE}"
    )


__all__ = ["reject_implicit_hf_conversion"]
