import numpy as np
import pytest

from radjax_student.legacy.losses import dense_kl_loss, sparse_topk_kl_loss


def test_sparse_topk_loss_matches_dense_loss_when_all_tokens_are_present() -> None:
    logits = np.asarray([[[1.0, 0.5, -0.5]]], dtype=np.float32)
    target_probs = np.asarray([[[0.2, 0.3, 0.5]]], dtype=np.float32)
    token_ids = np.asarray([[[0, 1, 2]]], dtype=np.int32)

    sparse_loss = sparse_topk_kl_loss(logits, token_ids, target_probs)
    dense_loss = dense_kl_loss(logits, target_probs)

    assert sparse_loss == pytest.approx(dense_loss)


def test_sparse_topk_loss_distributes_tail_mass_over_non_topk_tokens() -> None:
    logits = np.zeros((1, 1, 4), dtype=np.float32)
    token_ids = np.asarray([[[1, 3]]], dtype=np.int32)
    topk_probs = np.asarray([[[0.25, 0.25]]], dtype=np.float32)

    loss = sparse_topk_kl_loss(logits, token_ids, topk_probs, tail_mass=0.5)

    assert loss == pytest.approx(np.log(4.0))


def test_sparse_topk_loss_rejects_invalid_token_ids() -> None:
    logits = np.zeros((1, 1, 3), dtype=np.float32)
    token_ids = np.asarray([[[0, 3]]], dtype=np.int32)
    target_probs = np.asarray([[[0.5, 0.5]]], dtype=np.float32)

    with pytest.raises(ValueError, match="valid vocab indices"):
        sparse_topk_kl_loss(logits, token_ids, target_probs)
