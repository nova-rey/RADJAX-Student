"""The former NumPy training smoke, isolated from production exports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from radjax_contract.vocab import VocabContract

from radjax_student.debug.tiny_debug import TinyDebugStudentBackend
from radjax_student.legacy.losses import dense_kl_loss


@dataclass(frozen=True)
class TinyTrainStepResult:
    initial_loss: float
    final_loss: float
    parameters_changed: bool


def run_tiny_train_step() -> TinyTrainStepResult:
    backend = TinyDebugStudentBackend()
    vocab = VocabContract(tokenizer_id="toy", vocab_size=5)
    params = backend.init(backend.default_config(vocab_size=5), vocab, seed=0)
    input_ids = np.asarray([[0, 1, 2]], dtype=np.int32)
    teacher_probs = np.full((1, 3, 5), 0.2, dtype=np.float32)
    initial_logits = backend.forward(params, input_ids, train=True)
    initial_loss = dense_kl_loss(initial_logits, teacher_probs)
    updated = {name: value.copy() for name, value in params.items()}
    updated["head"] = updated["head"] - 0.01 * np.sign(updated["head"])
    final_logits = backend.forward(updated, input_ids, train=True)
    final_loss = dense_kl_loss(final_logits, teacher_probs)
    changed = any(not np.array_equal(params[name], updated[name]) for name in params)
    return TinyTrainStepResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        parameters_changed=changed,
    )


__all__ = ["TinyTrainStepResult", "run_tiny_train_step"]
