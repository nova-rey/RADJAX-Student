"""Passive optimizer backend interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
