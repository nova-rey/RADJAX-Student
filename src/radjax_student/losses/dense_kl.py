from __future__ import annotations

import numpy as np


def dense_kl_loss(student_logits: np.ndarray, teacher_probs: np.ndarray) -> float:
    logits = np.asarray(student_logits, dtype=np.float64)
    targets = np.asarray(teacher_probs, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    loss = -np.sum(targets * log_probs, axis=-1)
    return float(np.mean(loss))
