import numpy as np

from radjax_student.losses import dense_kl_loss
from radjax_student.training import run_tiny_train_step


def test_dense_kl_loss_computes_finite_value() -> None:
    logits = np.zeros((1, 2, 3), dtype=np.float32)
    teacher = np.full((1, 2, 3), 1.0 / 3.0, dtype=np.float32)

    loss = dense_kl_loss(logits, teacher)

    assert np.isfinite(loss)
    assert loss > 0.0


def test_one_tiny_train_step_moves_parameters() -> None:
    result = run_tiny_train_step()

    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
    assert result.parameters_changed is True
