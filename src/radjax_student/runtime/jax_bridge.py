"""Explicit runtime-owned conversion from key streams to JAX PRNG keys."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from radjax_student.runtime.keys import (
    JAX_KEY_BRIDGE_VERSION,
    RuntimeKeyStream,
    jax_key_words,
)


def derive_jax_key(
    stream: RuntimeKeyStream,
    *,
    global_step: int,
    micro_step: int,
    slot: str,
    invocation_index: int = 0,
) -> Any:
    """Create a JAX key from the versioned, caller-owned runtime identity."""

    jax = import_module("jax")
    jnp = import_module("jax.numpy")
    words = jnp.asarray(
        jax_key_words(
            stream,
            global_step=global_step,
            micro_step=micro_step,
            slot=slot,
            invocation_index=invocation_index,
        ),
        dtype=jnp.uint32,
    )
    return jax.random.wrap_key_data(words)


__all__ = ["JAX_KEY_BRIDGE_VERSION", "derive_jax_key"]
