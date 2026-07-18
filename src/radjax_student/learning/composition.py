"""Narrow application composition for the default learning operation set.

This module owns no architecture, objective, optimizer, or runtime defaults.
It is solely the application composition root that registers the declared
generic JAX learning operation with runtime's generic callable machinery.
"""

from __future__ import annotations

from radjax_student.runtime.callables import (
    RuntimeCallableRegistry,
    bind_runtime_callable,
)
from radjax_student.steps.jax_step import (
    GENERIC_JAX_LEARNING_STEP_DECLARATION,
    execute_jax_learning_step_kernel,
)


def build_default_learning_callable_registry() -> RuntimeCallableRegistry:
    """Build the explicit application registry for the generic learning step."""
    registry = RuntimeCallableRegistry()
    registry.register(
        bind_runtime_callable(
            callable=execute_jax_learning_step_kernel,
            declaration=GENERIC_JAX_LEARNING_STEP_DECLARATION,
        )
    )
    return registry


__all__ = ["build_default_learning_callable_registry"]
