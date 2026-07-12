"""Explicit JAX batch materialization; P3.11 defines finite-JSON test data only."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from radjax_student.learning.models import LearningBatch


@runtime_checkable
class JaxBatchMaterializer(Protocol):
    def materialize(self, batch: LearningBatch) -> Any: ...


class FiniteJsonJaxBatchMaterializer:
    """Test-only finite-JSON conversion, deliberately not a Tome payload bridge."""

    def materialize(self, batch: LearningBatch) -> Any:
        if not isinstance(batch, LearningBatch):
            raise TypeError("batch must be LearningBatch")
        from importlib import import_module

        from radjax_student.learning.jax_core import JaxBatch

        jnp = import_module("jax.numpy")
        return JaxBatch(
            inputs=_arrays(jnp, batch.inputs),
            targets=_arrays(jnp, batch.targets),
            weights=_arrays(jnp, batch.weights),
        )


def _arrays(jnp: Any, value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _arrays(jnp, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_arrays(jnp, item) for item in value)
    if isinstance(value, list):
        return jnp.asarray(value)
    return jnp.asarray(value)


__all__ = ["FiniteJsonJaxBatchMaterializer", "JaxBatchMaterializer"]
