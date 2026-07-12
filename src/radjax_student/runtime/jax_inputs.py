"""Runtime-owned placement and precision preparation for complete JAX inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class JaxPytreePlacer(Protocol):
    def place_execution_pytree(
        self, context: Any, value: Any, *, precision_policy: str
    ) -> Any: ...


@dataclass(frozen=True)
class PreparedJaxInputs:
    parameters: Any
    architecture_carry: Any
    optimizer_state: Any
    batch: Any
    metadata: dict[str, Any]


def prepare_jax_inputs(
    *,
    backend: JaxPytreePlacer,
    context: Any,
    parameters: Any,
    architecture_carry: Any,
    optimizer_state: Any,
    batch: Any,
    precision_policy: str,
) -> PreparedJaxInputs:
    """Place every complete-step input through one runtime policy boundary."""

    if precision_policy not in {
        "float32",
        "bfloat16",
        "float16",
        "mixed",
        "automatic",
        "unspecified",
    }:
        raise ValueError("unsupported runtime precision policy")
    placed = tuple(
        backend.place_execution_pytree(
            context, value, precision_policy=precision_policy
        )
        for value in (parameters, architecture_carry, optimizer_state, batch)
    )
    return PreparedJaxInputs(
        *placed,
        metadata={
            "placement_policy": context.metadata.get("placement_policy", "unspecified"),
            "precision_policy": precision_policy,
            "selected_device_id": context.metadata.get("selected_device_id"),
        },
    )


__all__ = ["JaxPytreePlacer", "PreparedJaxInputs", "prepare_jax_inputs"]
