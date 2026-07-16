"""Explicit runtime-owned conversion from key streams to JAX PRNG keys."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from radjax_student.runtime.keys import (
    JAX_KEY_BRIDGE_VERSION,
    RuntimeKeyStream,
    jax_key_words,
)

JAX_PRNG_IMPLEMENTATION = "threefry2x32"


class RuntimeJaxBridgeError(ValueError):
    """Stable ownership failure before a JAX key reaches dispatch."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def validate_runtime_jax_key_request(
    *,
    context: Any,
    stream: RuntimeKeyStream,
    global_step: int,
    micro_step: int,
    slot: str,
    invocation_index: int,
    expected_coordinates: Mapping[str, int] | None = None,
    prng_implementation: str = JAX_PRNG_IMPLEMENTATION,
) -> None:
    """Validate runtime-owned key identity before JAX import or dispatch.

    The complete learning step already derives its key from these coordinates.
    This narrow public boundary makes that ownership and its rejection reasons
    available to validation without exposing raw key words.
    """

    root_seed = getattr(context, "root_seed", None)
    if getattr(context, "backend_id", None) != "jax":
        raise RuntimeJaxBridgeError(
            "runtime_jax_backend_invalid",
            "JAX key derivation requires a selected JAX runtime context",
        )
    if stream.root_seed != root_seed:
        raise RuntimeJaxBridgeError(
            "runtime_jax_stream_ownership_invalid",
            "runtime key stream root seed does not belong to the context",
        )
    if prng_implementation != JAX_PRNG_IMPLEMENTATION:
        raise RuntimeJaxBridgeError(
            "runtime_jax_prng_implementation_invalid",
            "runtime JAX bridge requires the declared PRNG implementation",
        )
    try:
        jax_key_words(
            stream,
            global_step=global_step,
            micro_step=micro_step,
            slot=slot,
            invocation_index=invocation_index,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeJaxBridgeError(
            "runtime_jax_coordinates_invalid",
            "runtime JAX key coordinates are invalid",
        ) from exc
    if expected_coordinates is None:
        return
    for name, observed in (
        ("global_step", global_step),
        ("micro_step", micro_step),
        ("invocation_index", invocation_index),
    ):
        expected = expected_coordinates.get(name)
        if expected is not None and observed != expected:
            raise RuntimeJaxBridgeError(
                "runtime_jax_coordinates_mismatch",
                f"runtime JAX {name} does not match the learning transition",
            )


def validate_runtime_execution_evidence(value: Mapping[str, Any]) -> None:
    """Reject incomplete or contradictory JAX runtime receipt evidence."""

    required = {
        "backend_id",
        "mode",
        "compiled",
        "placement_policy",
        "precision_policy",
        "output_metadata_fields",
    }
    if set(value) != required:
        raise RuntimeJaxBridgeError(
            "runtime_jax_receipt_schema_invalid",
            "runtime receipt evidence fields are incomplete or unknown",
        )
    if value["backend_id"] != "jax":
        raise RuntimeJaxBridgeError(
            "runtime_jax_receipt_backend_invalid",
            "runtime receipt does not identify the selected JAX backend",
        )
    if not value["placement_policy"]:
        raise RuntimeJaxBridgeError(
            "runtime_jax_placement_missing",
            "runtime receipt must record placement policy",
        )
    if not value["precision_policy"]:
        raise RuntimeJaxBridgeError(
            "runtime_jax_precision_missing",
            "runtime receipt must record precision policy",
        )
    fields = value["output_metadata_fields"]
    if not isinstance(fields, (tuple, list)) or any(
        item in {"selected_device_id", "device_serial", "temporary_path", "timestamp"}
        for item in fields
    ):
        raise RuntimeJaxBridgeError(
            "runtime_jax_receipt_metadata_invalid",
            "runtime receipt exposes forbidden unstable metadata",
        )
    mode = value["mode"]
    compiled = value["compiled"]
    if mode == "eager" and compiled:
        raise RuntimeJaxBridgeError(
            "runtime_jax_receipt_mode_invalid",
            "eager execution cannot claim compilation",
        )
    if mode == "jit" and not compiled:
        raise RuntimeJaxBridgeError(
            "runtime_jax_receipt_mode_invalid",
            "JIT execution must report compilation",
        )
    if mode not in {"eager", "jit"} or not isinstance(compiled, bool):
        raise RuntimeJaxBridgeError(
            "runtime_jax_receipt_schema_invalid",
            "runtime mode and compiled fields are invalid",
        )


def validate_runtime_source_ownership(value: Mapping[str, str]) -> None:
    """Reject architecture/optimizer/learning source that owns runtime controls."""

    owner = value.get("owner")
    source = value.get("source")
    if not isinstance(owner, str) or not isinstance(source, str):
        raise RuntimeJaxBridgeError(
            "runtime_jax_source_invalid", "runtime ownership source is malformed"
        )
    forbidden = {
        "architecture": ("jax.devices(", "jax.jit("),
        "optimizer": ("jax.devices(", "jax.jit("),
        "learning": ("jax.jit(",),
    }
    if any(token in source for token in forbidden.get(owner, ())):
        raise RuntimeJaxBridgeError(
            "runtime_jax_ownership_violation",
            "non-runtime source attempted to own JAX placement or compilation",
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
    return jax.random.wrap_key_data(words, impl=JAX_PRNG_IMPLEMENTATION)


__all__ = [
    "JAX_KEY_BRIDGE_VERSION",
    "JAX_PRNG_IMPLEMENTATION",
    "RuntimeJaxBridgeError",
    "derive_jax_key",
    "validate_runtime_execution_evidence",
    "validate_runtime_jax_key_request",
    "validate_runtime_source_ownership",
]
