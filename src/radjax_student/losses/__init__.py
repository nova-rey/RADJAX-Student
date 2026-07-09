"""Student loss functions."""

from radjax_student.losses.dense_kl import dense_kl_loss
from radjax_student.losses.sparse_topk import sparse_topk_kl_loss

__all__ = ["dense_kl_loss", "sparse_topk_kl_loss"]
