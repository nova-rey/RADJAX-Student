"""Passive interfaces between generic learning and concrete model math."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from radjax_student.architecture.models import (
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureInitRequest,
    ArchitectureInitResult,
    ArchitectureMetadata,
    BatchValidationResult,
    ForwardRequest,
    ForwardResult,
    ParameterCatalog,
    ResolvedObjectiveSelection,
)
from radjax_student.contracts import (
    HFCompatibilityDescriptor,
    LearningBatch,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)


@runtime_checkable
class ArchitecturePlugin(Protocol):
    """Architecture-owned math and parameter-meaning boundary."""

    architecture_id: str
    architecture_version: int

    def capability_profile(self) -> ArchitectureCapabilityProfile: ...

    def validate_config(self, config: ArchitectureConfig) -> None: ...

    def initialize_parameters(
        self, request: ArchitectureInitRequest
    ) -> ArchitectureInitResult: ...

    def describe_parameters(
        self, parameters: object | None = None
    ) -> ParameterCatalog: ...

    def architecture_metadata(self) -> ArchitectureMetadata: ...

    def validate_batch(
        self, batch: LearningBatch, config: ArchitectureConfig
    ) -> BatchValidationResult: ...

    def forward(self, request: ForwardRequest) -> ForwardResult: ...

    def resolve_update_scope(
        self, scope: UpdateScope, parameter_catalog: ParameterCatalog
    ) -> ResolvedUpdateSelection: ...

    def resolve_objective_scope(
        self, scope: ObjectiveScope, metadata: ArchitectureMetadata
    ) -> ResolvedObjectiveSelection: ...

    def hf_compatibility_descriptor(
        self,
        request: ArchitectureInitRequest,
        result: ArchitectureInitResult,
    ) -> HFCompatibilityDescriptor: ...


@runtime_checkable
class JaxArchitectureExecution(Protocol):
    """Optional JAX capability of an existing architecture plugin identity."""

    def apply_jax(
        self,
        parameters: Any,
        architecture_state: Any,
        batch: Any,
        *,
        objective_scope: ObjectiveScope,
        training: bool,
        rng_key: Any | None,
    ) -> ForwardResult: ...


@runtime_checkable
class JaxArchitecturePlugin(ArchitecturePlugin, JaxArchitectureExecution, Protocol):
    """The only production JAX architecture identity."""
