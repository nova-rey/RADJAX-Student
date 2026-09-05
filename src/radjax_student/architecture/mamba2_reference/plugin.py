"""Static Mamba-2 reference plugin boundary (M2.2).

The package intentionally advertises inspection and validation only until the
initialization and pure-JAX execution checkpoints are accepted.  Keeping the
identity registered at this stage lets callers fail closed on unsupported
capabilities instead of mistaking a partial plugin for an executable model.
"""

from __future__ import annotations

from dataclasses import dataclass

from radjax_student.architecture.errors import (
    ArchitectureContractError,
    ArchitectureIssue,
)
from radjax_student.architecture.mamba2_reference.config import (
    MAMBA2_REFERENCE_ARCHITECTURE_ID,
    MAMBA2_REFERENCE_ARCHITECTURE_VERSION,
    reference_architecture_config,
    validate_mamba2_config,
)
from radjax_student.architecture.mamba2_reference.schema import (
    architecture_metadata,
    capability_profile,
    hf_descriptor,
    parameter_catalog,
    parameter_layout,
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
from radjax_student.contracts import (
    HFCompatibilityDescriptor,
    LearningBatch,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)


@dataclass(frozen=True)
class Mamba2ReferencePlugin:
    """Explicit Mamba-2 identity with static-only M2.2 capability."""

    architecture_id: str = MAMBA2_REFERENCE_ARCHITECTURE_ID
    architecture_version: int = MAMBA2_REFERENCE_ARCHITECTURE_VERSION

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return capability_profile()

    def validate_config(self, config: ArchitectureConfig) -> None:
        validate_mamba2_config(config)

    def describe_parameters(
        self,
        parameters: object | None = None,
        *,
        architecture_config: ArchitectureConfig | None = None,
    ) -> ParameterCatalog:
        config = (
            reference_architecture_config()
            if architecture_config is None
            else architecture_config
        )
        self.validate_config(config)
        catalog = parameter_catalog(config)
        if parameters is not None:
            try:
                parameter_layout(config).validate_materialized_parameters(parameters)
            except (TypeError, ValueError) as exc:
                raise ArchitectureContractError(
                    "architecture_parameter_catalog_invalid",
                    "materialized parameters do not match the Mamba-2 static layout",
                ) from exc
        return catalog

    def architecture_metadata(self) -> ArchitectureMetadata:
        return architecture_metadata()

    def initialize_parameters(
        self, request: ArchitectureInitRequest
    ) -> ArchitectureInitResult:
        del request
        raise ArchitectureContractError(
            "architecture_initialization_failed",
            "Mamba-2 initialization is unavailable before M2.3",
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
        valid = (
            isinstance(token_ids, (list, tuple))
            and len(token_ids) == 1
            and isinstance(token_ids[0], (list, tuple))
            and 1 <= len(token_ids[0]) <= config.sequence_length
            and all(
                isinstance(token, int)
                and not isinstance(token, bool)
                and 0 <= token < config.vocab_size
                for token in token_ids[0]
            )
        )
        if not valid:
            return BatchValidationResult(
                status="fail",
                blockers=(
                    ArchitectureIssue(
                        code="architecture_batch_incompatible",
                        message=(
                            "Mamba-2 requires one rank-2 integer token sequence "
                            "whose length does not exceed the configured context"
                        ),
                    ),
                ),
            )
        return BatchValidationResult(status="pass")

    def forward(self, request: ForwardRequest) -> ForwardResult:
        del request
        raise ArchitectureContractError(
            "architecture_forward_failed",
            "Mamba-2 JAX execution is unavailable before M2.4",
        )

    def resolve_update_scope(
        self, scope: UpdateScope, parameter_catalog: ParameterCatalog
    ) -> ResolvedUpdateSelection:
        if parameter_catalog.architecture_id != self.architecture_id:
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "parameter catalog does not match the Mamba-2 static schema",
            )
        if scope.kind == "whole_student":
            selected = parameter_catalog.trainable_paths
        elif scope.kind == "parameter_paths":
            selected = scope.parameter_paths
            unknown = sorted(set(selected) - set(parameter_catalog.paths))
            if unknown:
                raise ArchitectureContractError(
                    "architecture_parameter_path_unknown",
                    "update scope references an unknown Mamba-2 parameter path",
                    details={"unknown_paths": unknown},
                )
        elif scope.kind == "named_region":
            region_id = scope.region_id
            region_paths = {
                path
                for path in parameter_catalog.paths
                if region_id in parameter_catalog.get(path).region_ids
            }
            if not region_paths:
                raise ArchitectureContractError(
                    "architecture_parameter_path_unknown",
                    "update scope references an unknown Mamba-2 named region",
                    details={"region_id": region_id},
                )
            selected = tuple(
                path for path in parameter_catalog.paths if path in region_paths
            )
        else:
            raise ArchitectureContractError(
                "architecture_update_scope_unsupported",
                "Mamba-2 supports whole-student, named-region, and explicit-path "
                "scopes",
            )
        return ResolvedUpdateSelection(
            selection_id=f"{self.architecture_id}:{scope.kind}",
            selected_parameter_paths=tuple(selected),
            excluded_parameter_paths=tuple(
                path for path in parameter_catalog.paths if path not in selected
            ),
            capabilities=(f"architecture.update_scope.{scope.kind}_v1",),
            metadata={"phase": "M2.2", "jax_execution_available": False},
        )

    def resolve_objective_scope(
        self, scope: ObjectiveScope, metadata: ArchitectureMetadata
    ) -> ResolvedObjectiveSelection:
        if (
            metadata.architecture_id != self.architecture_id
            or scope.kind != "final_output"
        ):
            raise ArchitectureContractError(
                "architecture_objective_scope_unsupported",
                "Mamba-2 exposes only the final logits objective surface",
            )
        return ResolvedObjectiveSelection(
            scope=scope,
            surface_id="final_output",
            surface_role="logits",
            required_capabilities=("architecture.objective.final_output_v1",),
            metadata={"phase": "M2.2", "jax_execution_available": False},
        )

    def hf_compatibility_descriptor(
        self, request: ArchitectureInitRequest, result: ArchitectureInitResult
    ) -> HFCompatibilityDescriptor:
        self.validate_config(request.config)
        expected_catalog = parameter_catalog(request.config)
        expected_layout = parameter_layout(request.config)
        if (
            result.parameter_catalog != expected_catalog
            or result.parameter_layout != expected_layout
        ):
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "HF projection must use the declared Mamba-2 static schema",
            )
        return hf_descriptor(request.config)


__all__ = ["Mamba2ReferencePlugin"]
