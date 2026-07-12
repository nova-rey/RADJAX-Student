"""A non-numerical architecture test double used to prove P3.2 contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from radjax_student.architecture.errors import (
    ArchitectureContractError,
    ArchitectureIssue,
)
from radjax_student.architecture.models import (
    ARCHITECTURE_CLAIMS_NOT_MADE,
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureInitRequest,
    ArchitectureInitResult,
    ArchitectureMetadata,
    ArchitectureState,
    BatchValidationResult,
    ForwardRequest,
    ForwardResult,
    IntermediateSurfaceDescriptor,
    NamedRegion,
    ParameterCatalog,
    ParameterDescriptor,
    ResolvedObjectiveSelection,
)
from radjax_student.contracts import (
    LearningBatch,
    ObjectiveScope,
    ResolvedUpdateSelection,
    UpdateScope,
)

FAKE_ARCHITECTURE_ID = "test.architecture.v1"
FAKE_ARCHITECTURE_CAPABILITIES: tuple[str, ...] = (
    "architecture.batch_validation_v1",
    "architecture.forward_v1",
    "architecture.initialize_parameters_v1",
    "architecture.objective.final_output_v1",
    "architecture.objective.intermediate_surface_v1",
    "architecture.parameter_metadata_v1",
    "architecture.update_scope.named_region_v1",
    "architecture.update_scope.parameter_paths_v1",
    "architecture.update_scope.whole_student_v1",
)


@dataclass(frozen=True)
class TestParameterTree:
    """Opaque test-only parameter identity; it contains no numerical values."""

    paths: tuple[str, ...]


@dataclass(frozen=True)
class TestForwardOutputs:
    """Opaque test-only output identity; it contains no numerical values."""

    batch_id: str


@dataclass(frozen=True)
class FakeArchitecturePlugin:
    """Contract test double, not a concrete Student model implementation."""

    architecture_id: str = FAKE_ARCHITECTURE_ID
    architecture_version: int = 1

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return ArchitectureCapabilityProfile(
            architecture_id=self.architecture_id,
            version=self.architecture_version,
            capabilities=FAKE_ARCHITECTURE_CAPABILITIES,
            metadata={"test_double": True},
        )

    def validate_config(self, config: ArchitectureConfig) -> None:
        if config.architecture_id != self.architecture_id:
            raise ArchitectureContractError(
                "architecture_config_invalid",
                "configuration architecture ID does not match plugin",
                details={
                    "expected": self.architecture_id,
                    "received": config.architecture_id,
                },
            )

    def describe_parameters(self, parameters: object | None = None) -> ParameterCatalog:
        del parameters
        return ParameterCatalog(
            architecture_id=self.architecture_id,
            parameters=(
                ParameterDescriptor(
                    path="trunk.weight",
                    shape=(4, 4),
                    dtype="float32",
                    role="recurrent_block",
                    region_ids=("trunk", "shared", "whole_student"),
                ),
                ParameterDescriptor(
                    path="trunk.bias",
                    shape=(4,),
                    dtype="float32",
                    role="recurrent_block",
                    region_ids=("trunk", "whole_student"),
                ),
                ParameterDescriptor(
                    path="head.weight",
                    shape=(4, 4),
                    dtype="float32",
                    role="output_head",
                    region_ids=("head", "shared", "whole_student"),
                ),
            ),
            metadata={"test_double": True},
        )

    def architecture_metadata(self) -> ArchitectureMetadata:
        catalog = self.describe_parameters()
        return ArchitectureMetadata(
            architecture_id=self.architecture_id,
            parameter_catalog=catalog,
            capability_profile=self.capability_profile(),
            named_regions=(
                NamedRegion("whole_student", catalog.trainable_paths),
                NamedRegion("trunk", ("trunk.weight", "trunk.bias")),
                NamedRegion("head", ("head.weight",)),
                NamedRegion("shared", ("trunk.weight", "head.weight")),
            ),
            objective_surfaces=(
                IntermediateSurfaceDescriptor(
                    surface_id="final_output",
                    kind="logits",
                    available_in_training=True,
                    available_in_inference=True,
                ),
                IntermediateSurfaceDescriptor(
                    surface_id="trunk_output",
                    kind="hidden_state",
                    region_id="trunk",
                    available_in_training=True,
                    available_in_inference=True,
                ),
            ),
            warnings=(
                ArchitectureIssue(
                    code="architecture_named_regions_overlap",
                    message=(
                        "The test double intentionally declares overlapping regions."
                    ),
                    details={"regions": ["shared", "trunk", "head"]},
                ),
                ArchitectureIssue(
                    code="architecture_plugin_test_double",
                    message=(
                        "This plugin proves contracts only and implements no model "
                        "math."
                    ),
                ),
            ),
        )

    def initialize_parameters(
        self, request: ArchitectureInitRequest
    ) -> ArchitectureInitResult:
        self.validate_config(request.config)
        catalog = self.describe_parameters()
        return ArchitectureInitResult(
            parameter_catalog=catalog,
            architecture_state=ArchitectureState(
                state_id=f"{self.architecture_id}:initial",
                metadata={"runtime_keys_reference": request.runtime_keys_reference},
            ),
            parameters=TestParameterTree(catalog.paths),
            warnings=(
                ArchitectureIssue(
                    code="architecture_plugin_test_double",
                    message=(
                        "Test parameter identities were declared without numerical "
                        "initialization."
                    ),
                ),
            ),
        )

    def validate_batch(
        self, batch: LearningBatch, config: ArchitectureConfig
    ) -> BatchValidationResult:
        self.validate_config(config)
        token_ids = batch.inputs.get("token_ids")
        if not isinstance(token_ids, Mapping):
            return _batch_failure(
                "token_ids", "missing required token_ids shape metadata"
            )
        rank = token_ids.get("rank")
        sequence_length = token_ids.get("sequence_length")
        if rank != 2:
            return _batch_failure("token_ids.rank", "token_ids rank must be 2")
        if not isinstance(sequence_length, int) or sequence_length < 1:
            return _batch_failure(
                "token_ids.sequence_length",
                "sequence length must be a positive integer",
            )
        if (
            config.sequence_length is not None
            and sequence_length > config.sequence_length
        ):
            return _batch_failure(
                "token_ids.sequence_length",
                "sequence length exceeds architecture configuration",
            )
        return BatchValidationResult(
            status="pass",
            metadata={
                "checked_fields": ["token_ids.rank", "token_ids.sequence_length"]
            },
        )

    def forward(self, request: ForwardRequest) -> ForwardResult:
        return ForwardResult(
            outputs=TestForwardOutputs(request.batch.batch_id),
            intermediate_surfaces=(
                "final_output",
                "trunk_output",
            )
            if request.objective_scope.kind == "intermediate_surface"
            else ("final_output",),
            output_metadata={"test_double": True, "batch_id": request.batch.batch_id},
            warnings=(
                ArchitectureIssue(
                    code="architecture_plugin_test_double",
                    message="Forward output is a non-numerical contract token.",
                ),
            ),
        )

    def resolve_update_scope(
        self, scope: UpdateScope, parameter_catalog: ParameterCatalog
    ) -> ResolvedUpdateSelection:
        if parameter_catalog.architecture_id != self.architecture_id:
            raise ArchitectureContractError(
                "architecture_parameter_catalog_invalid",
                "parameter catalog does not belong to this plugin",
            )
        metadata = self.architecture_metadata()
        if scope.kind == "whole_student":
            paths = parameter_catalog.trainable_paths
        elif scope.kind == "named_region":
            assert scope.region_id is not None
            try:
                paths = tuple(
                    path
                    for path in metadata.region(scope.region_id).parameter_paths
                    if parameter_catalog.get(path).trainable_by_default
                )
            except KeyError as exc:
                raise ArchitectureContractError(
                    "architecture_update_scope_resolution_failed",
                    "named update region is not declared by this architecture",
                    details={"region_id": scope.region_id},
                ) from exc
        elif scope.kind == "parameter_paths":
            paths = scope.parameter_paths
            for path in paths:
                try:
                    descriptor = parameter_catalog.get(path)
                except KeyError as exc:
                    raise ArchitectureContractError(
                        "architecture_parameter_path_unknown",
                        "update scope references an unknown parameter path",
                        details={"path": path},
                    ) from exc
                if not descriptor.trainable_by_default:
                    raise ArchitectureContractError(
                        "architecture_update_scope_resolution_failed",
                        "update scope references a parameter that is not trainable "
                        "by default",
                        details={"path": path},
                    )
        else:
            raise ArchitectureContractError(
                "architecture_update_scope_unsupported",
                "the test plugin does not define plugin-defined update scopes",
            )
        selected = tuple(sorted(paths))
        return ResolvedUpdateSelection(
            selection_id=f"{self.architecture_id}:{scope.kind}:{','.join(selected)}",
            selected_parameter_paths=selected,
            excluded_parameter_paths=tuple(
                path for path in parameter_catalog.paths if path not in selected
            ),
            capabilities=(f"architecture.update_scope.{scope.kind}_v1",),
            metadata={
                "architecture_id": self.architecture_id,
                "claims_not_made": list(ARCHITECTURE_CLAIMS_NOT_MADE),
                "warnings": ["architecture_scope_resolution_declaration_only"],
            },
        )

    def resolve_objective_scope(
        self, scope: ObjectiveScope, metadata: ArchitectureMetadata
    ) -> ResolvedObjectiveSelection:
        if metadata.architecture_id != self.architecture_id:
            raise ArchitectureContractError(
                "architecture_objective_scope_resolution_failed",
                "architecture metadata does not belong to this plugin",
            )
        if scope.kind == "final_output":
            surface_id = "final_output"
        elif scope.kind == "intermediate_surface":
            assert scope.target_id is not None
            surface_id = scope.target_id
        else:
            raise ArchitectureContractError(
                "architecture_objective_scope_unsupported",
                "the requested objective scope is not declared by this plugin",
                details={"kind": scope.kind, "target_id": scope.target_id},
            )
        try:
            metadata.surface(surface_id)
        except KeyError as exc:
            raise ArchitectureContractError(
                "architecture_objective_scope_resolution_failed",
                "objective surface is not declared by this architecture",
                details={"surface_id": surface_id},
            ) from exc
        return ResolvedObjectiveSelection(
            scope=scope,
            surface_id=surface_id,
            required_capabilities=(f"architecture.objective.{scope.kind}_v1",),
            metadata={"architecture_id": self.architecture_id},
        )


def _batch_failure(field: str, message: str) -> BatchValidationResult:
    return BatchValidationResult(
        status="fail",
        blockers=(
            ArchitectureIssue(
                code="architecture_batch_incompatible",
                message=message,
                details={"field": field},
            ),
        ),
    )
