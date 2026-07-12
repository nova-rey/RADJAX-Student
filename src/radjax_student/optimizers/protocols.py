"""Passive optimizer backend interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from radjax_student.contracts import JaxOptimizerStateDescriptor, ParameterTreeLayout
from radjax_student.optimizers.models import (
    OptimizerCapabilityProfile,
    OptimizerConfig,
    OptimizerInitRequest,
    OptimizerInitResult,
    OptimizerState,
    OptimizerStateDescriptor,
    OptimizerUpdateRequest,
    OptimizerUpdateResult,
)


@runtime_checkable
class OptimizerBackend(Protocol):
    optimizer_id: str
    optimizer_version: int

    def capability_profile(self) -> OptimizerCapabilityProfile: ...
    def validate_config(self, config: OptimizerConfig) -> None: ...
    def initialize_state(
        self, request: OptimizerInitRequest
    ) -> OptimizerInitResult: ...
    def apply_updates(
        self, request: OptimizerUpdateRequest
    ) -> OptimizerUpdateResult: ...
    def describe_state(self, state: OptimizerState) -> OptimizerStateDescriptor: ...


@runtime_checkable
class JaxOptimizerExecution(Protocol):
    """Optional JAX capability attached to the existing optimizer identity."""

    def jax_state_descriptor(
        self, parameter_layout: ParameterTreeLayout
    ) -> JaxOptimizerStateDescriptor: ...

    def initialize_jax_state(
        self,
        *,
        config: OptimizerConfig,
        parameter_layout: ParameterTreeLayout,
        optimizer_state: OptimizerState,
    ) -> Any: ...

    def validate_jax_state(
        self,
        *,
        arrays: Any,
        descriptor: JaxOptimizerStateDescriptor,
    ) -> None: ...

    def apply_jax_updates(
        self,
        *,
        parameters: Any,
        gradients: Any,
        optimizer_array_state: Any,
        update_mask: Any,
        config: OptimizerConfig,
        schedule_values: dict[str, Any],
    ) -> tuple[Any, Any, Any, dict[str, Any]]: ...


@runtime_checkable
class JaxOptimizerBackend(OptimizerBackend, JaxOptimizerExecution, Protocol):
    """The only production JAX optimizer identity."""
