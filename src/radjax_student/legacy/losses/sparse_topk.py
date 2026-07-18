"""Legacy NumPy sparse-top-k analysis implementation; not a training objective."""

from __future__ import annotations

import numpy as np


def sparse_topk_kl_loss(
    student_logits: np.ndarray,
    target_token_ids: np.ndarray,
    target_probs: np.ndarray,
    *,
    tail_mass: np.ndarray | float | None = None,
) -> float:
    logits = np.asarray(student_logits, dtype=np.float64)
    token_ids = np.asarray(target_token_ids, dtype=np.int64)
    probs = np.asarray(target_probs, dtype=np.float64)
    _validate_sparse_shapes(logits, token_ids, probs)

    log_probs = _log_softmax(logits)
    gathered = np.take_along_axis(log_probs, token_ids, axis=-1)
    loss = -np.sum(probs * gathered, axis=-1)
    if tail_mass is not None:
        loss = loss + _tail_loss(log_probs, gathered, np.asarray(tail_mass))
    return float(np.mean(loss))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _validate_sparse_shapes(
    logits: np.ndarray,
    token_ids: np.ndarray,
    probs: np.ndarray,
) -> None:
    if logits.ndim < 1:
        raise ValueError("student_logits must have a vocab dimension")
    if token_ids.shape != probs.shape:
        raise ValueError("target_token_ids and target_probs must have the same shape")
    if token_ids.shape[:-1] != logits.shape[:-1]:
        raise ValueError("target batch dimensions must match student_logits")
    if token_ids.shape[-1] > logits.shape[-1]:
        raise ValueError("top-k width cannot exceed vocab size")
    if token_ids.size and (
        np.min(token_ids) < 0 or np.max(token_ids) >= logits.shape[-1]
    ):
        raise ValueError("target_token_ids must be valid vocab indices")
    if np.any(probs < 0.0) or not np.all(np.isfinite(probs)):
        raise ValueError("target_probs must be finite non-negative values")
    if np.any(np.sum(probs, axis=-1) > 1.0 + 1e-6):
        raise ValueError("target_probs may not sum to more than 1 per position")


def _tail_loss(
    log_probs: np.ndarray,
    gathered_log_probs: np.ndarray,
    tail_mass: np.ndarray,
) -> np.ndarray:
    normalized_tail = np.asarray(tail_mass, dtype=np.float64)
    if normalized_tail.shape == ():
        normalized_tail = np.full(log_probs.shape[:-1], float(normalized_tail))
    if normalized_tail.shape != log_probs.shape[:-1]:
        raise ValueError("tail_mass must be scalar or match target batch dimensions")
    if np.any(normalized_tail < 0.0) or not np.all(np.isfinite(normalized_tail)):
        raise ValueError("tail_mass must be finite and non-negative")

    tail_count = log_probs.shape[-1] - gathered_log_probs.shape[-1]
    if tail_count <= 0 and np.any(normalized_tail > 0.0):
        raise ValueError("positive tail_mass requires at least one non-top-k token")
    if tail_count <= 0:
        return np.zeros(log_probs.shape[:-1], dtype=np.float64)

    topk_log_prob_sum = np.sum(gathered_log_probs, axis=-1)
    tail_log_prob_sum = np.sum(log_probs, axis=-1) - topk_log_prob_sum
    return -(normalized_tail / float(tail_count)) * tail_log_prob_sum
