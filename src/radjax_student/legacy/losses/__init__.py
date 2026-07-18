"""Noncanonical NumPy losses retained only for legacy and offline analysis."""

from radjax_student.legacy.losses.dense_kl import dense_kl_loss
from radjax_student.legacy.losses.sparse_topk import sparse_topk_kl_loss

__all__ = ["dense_kl_loss", "sparse_topk_kl_loss"]
