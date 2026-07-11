"""Passive interfaces between generic learning and concrete model math."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
from radjax_student.learning import (
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
