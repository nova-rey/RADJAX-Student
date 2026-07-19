"""Static-only RWKV-7 reference architecture plugin for P4.2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from radjax_student.architecture.errors import (
    ArchitectureContractError,
    ArchitectureIssue,
)
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
from radjax_student.architecture.rwkv7_reference.config import (
    RWKV7_REFERENCE_ARCHITECTURE_ID,
    RWKV7_REFERENCE_ARCHITECTURE_VERSION,
    validate_reference_config,
)
from radjax_student.architecture.rwkv7_reference.schema import (
    architecture_metadata,
    capability_profile,
    hf_descriptor,
    parameter_catalog,
    parameter_layout,
)
from radjax_student.contracts import (
    HFCompatibilityDescriptor,
    LearningBatch,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)


@dataclass(frozen=True)
class RWKV7ReferencePlugin:
    """A complete static contract implementation, deliberately not JAX executable."""

    architecture_id: str = RWKV7_REFERENCE_ARCHITECTURE_ID
    architecture_version: int = RWKV7_REFERENCE_ARCHITECTURE_VERSION

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return capability_profile()

    def validate_config(self, config: ArchitectureConfig) -> None:
        validate_reference_config(config)

    def describe_parameters(self, parameters: object | None = None) -> ParameterCatalog:
        if parameters is not None:
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "P4.2 schema does not accept materialized parameter values",
            )
        return parameter_catalog()

    def architecture_metadata(self) -> ArchitectureMetadata:
        return architecture_metadata()

    def initialize_parameters(
        self, request: ArchitectureInitRequest
    ) -> ArchitectureInitResult:
        self.validate_config(request.config)
        raise ArchitectureContractError(
            "architecture_initialization_failed",
            "RWKV-7 parameter initialization is unavailable before P4.3",
        )

    def validate_batch(
        self, batch: LearningBatch, config: ArchitectureConfig
    ) -> BatchValidationResult:
        self.validate_config(config)
        if not isinstance(batch, LearningBatch):
            raise ArchitectureContractError(
                "architecture_batch_incompatible", "batch must be LearningBatch"
            )
        token_ids = batch.inputs.get("token_ids")
        if not isinstance(token_ids, Mapping) or token_ids.get("rank") != 2:
            return BatchValidationResult(
                status="fail",
                blockers=(
                    ArchitectureIssue(
                        code="architecture_batch_incompatible",
                        message="RWKV-7 reference expects rank-2 token_ids metadata",
                    ),
                ),
            )
        return BatchValidationResult(status="pass")

    def forward(self, request: ForwardRequest) -> ForwardResult:
        del request
        raise ArchitectureContractError(
            "architecture_forward_failed",
            "RWKV-7 forward execution is unavailable before P4.4",
        )

    def resolve_update_scope(
        self, scope: UpdateScope, parameter_catalog: ParameterCatalog
    ) -> ResolvedUpdateSelection:
        if parameter_catalog != self.describe_parameters():
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "parameter catalog does not match the RWKV-7 static schema",
            )
        if scope.kind == "whole_student":
            selected = parameter_catalog.trainable_paths
        elif scope.kind == "parameter_paths":
            selected = scope.parameter_paths
            unknown = sorted(set(selected) - set(parameter_catalog.paths))
            if unknown:
                raise ArchitectureContractError(
                    "architecture_parameter_path_unknown",
                    "update scope references an unknown RWKV-7 parameter path",
                    details={"unknown_paths": unknown},
                )
        else:
            raise ArchitectureContractError(
                "architecture_update_scope_unsupported",
                "P4.2 supports only whole-student or explicit-path static scopes",
            )
        return ResolvedUpdateSelection(
            selection_id=f"{self.architecture_id}:{scope.kind}",
            selected_parameter_paths=tuple(selected),
            excluded_parameter_paths=tuple(
                path for path in parameter_catalog.paths if path not in selected
            ),
            capabilities=(f"architecture.update_scope.{scope.kind}_v1",),
            metadata={"phase": "P4.2", "static_only": True},
        )

    def resolve_objective_scope(
        self, scope: ObjectiveScope, metadata: ArchitectureMetadata
    ) -> ResolvedObjectiveSelection:
        if metadata != self.architecture_metadata() or scope.kind != "final_output":
            raise ArchitectureContractError(
                "architecture_objective_scope_unsupported",
                "P4.2 declares only the final logits objective surface",
            )
        return ResolvedObjectiveSelection(
            scope=scope,
            surface_id="final_output",
            surface_role="logits",
            required_capabilities=("architecture.objective.final_output_v1",),
            metadata={"phase": "P4.2", "static_only": True},
        )

    def hf_compatibility_descriptor(
        self, request: ArchitectureInitRequest, result: ArchitectureInitResult
    ) -> HFCompatibilityDescriptor:
        self.validate_config(request.config)
        if (
            result.parameter_catalog != parameter_catalog()
            or result.parameter_layout != parameter_layout()
        ):
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "HF projection must use the declared RWKV-7 static schema",
            )
        return hf_descriptor(request.config)


__all__ = ["RWKV7ReferencePlugin"]
