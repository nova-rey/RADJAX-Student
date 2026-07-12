"""Typed JAX optimizer-state envelope without importing JAX at module import."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from radjax_student.contracts import JaxOptimizerStateDescriptor, ParameterTreeLayout
from radjax_student.optimizers.errors import OptimizerContractError
from radjax_student.optimizers.models import OptimizerState


@dataclass(frozen=True)
class JaxOptimizerState:
    """Stable optimizer identity plus opaque algorithm-owned numerical leaves."""

    envelope: OptimizerState
    descriptor: JaxOptimizerStateDescriptor
    arrays: Any

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, OptimizerState):
            raise TypeError("envelope must be OptimizerState")
        if not isinstance(self.descriptor, JaxOptimizerStateDescriptor):
            raise TypeError("descriptor must be JaxOptimizerStateDescriptor")
        if self.envelope.optimizer_id != self.descriptor.optimizer_id:
            raise OptimizerContractError(
                "optimizer_jax_state_invalid",
                "JAX optimizer state descriptor does not match optimizer identity",
            )


def validate_jax_optimizer_state(
    state: JaxOptimizerState,
    *,
    optimizer_id: str,
    parameter_layout: ParameterTreeLayout,
    descriptor: JaxOptimizerStateDescriptor,
) -> None:
    """Reject identity, capability, schema, or layout mismatches before execution."""

    if not isinstance(state, JaxOptimizerState):
        raise OptimizerContractError(
            "optimizer_jax_state_invalid", "JAX optimizer state is required"
        )
    if state.envelope.optimizer_id != optimizer_id:
        raise OptimizerContractError(
            "optimizer_jax_state_invalid",
            "optimizer state belongs to a different optimizer",
        )
    if state.descriptor != descriptor:
        raise OptimizerContractError(
            "optimizer_jax_state_invalid",
            "optimizer capability, schema, or state layout does not match",
        )
    if state.descriptor.layout_digest != parameter_layout.digest():
        raise OptimizerContractError(
            "optimizer_jax_state_invalid",
            "optimizer state layout digest does not match parameters",
        )


def advanced_jax_optimizer_state(
    state: JaxOptimizerState, arrays: Any
) -> JaxOptimizerState:
    """Advance only the transport envelope; algorithm leaves remain opaque."""

    return JaxOptimizerState(
        envelope=OptimizerState(
            optimizer_id=state.envelope.optimizer_id,
            parameter_paths=state.envelope.parameter_paths,
            step=state.envelope.step + 1,
            schema_version=state.envelope.schema_version,
            state_structure=state.envelope.state_structure,
            backend_state=state.envelope.backend_state,
            metadata=state.envelope.metadata,
            claims_not_made=state.envelope.claims_not_made,
        ),
        descriptor=state.descriptor,
        arrays=arrays,
    )


def require_finite_jax_gradients(metrics: Mapping[str, Any]) -> None:
    """Turn the pure update's finite flag into a stable boundary failure."""

    finite = metrics.get("gradients_finite")
    if finite is None or not bool(finite):
        raise OptimizerContractError(
            "optimizer_gradient_nonfinite", "JAX gradients must be finite"
        )


__all__ = [
    "JaxOptimizerState",
    "advanced_jax_optimizer_state",
    "require_finite_jax_gradients",
    "validate_jax_optimizer_state",
]
