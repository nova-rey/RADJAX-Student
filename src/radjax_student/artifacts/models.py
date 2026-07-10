from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from radjax_contract.tome import (
    TomeCompression,
    TomeCoverPage,
    TomeManifest,
    TomePayloadFormat,
)
from radjax_contract.tome.production import (
    ArtifactLocalFingerprintId,
    ArtifactLocalModeId,
    ProductionTomeContentRef,
    RecommendedTrainingPlan,
)
from radjax_contract.vocab import VocabContract


@dataclass(frozen=True)
class TomePayloadSummary:
    """Legacy dense-v0 payload summary retained for smoke/debug callers."""

    payload_format: TomePayloadFormat
    compression: TomeCompression
    expected_adapter: str | None
    implemented_by_contract: bool
    record_count: int | None
    sequence_length: int | None
    shard_count: int
    shard_paths: tuple[str, ...]


@dataclass(frozen=True)
class TomeInferredDefaults:
    """Provisional P1.2 seed values; production correction belongs to P1.7."""

    role: str | None
    teacher_id: str | None
    teacher_family: str | None
    teacher_backend: str | None
    tokenizer_id: str | None
    vocab_size: int | None
    adapter_family: str | None
    compression_family: str | None
    requires_reconstruction: bool | None


@dataclass(frozen=True)
class TomeArtifactIdentity:
    artifact_kind: str
    cover_page_version: int | str
    tome_version: int | None
    layout: str
    source_artifact_type: str
    created_by: str | None
    created_at: str | None
    producer_identity: str


@dataclass(frozen=True)
class TomeArtifactProvenance:
    teacher: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    targets: Mapping[str, Any]
    corpus: Mapping[str, Any] | None
    teacher_model: Mapping[str, Any] | None
    producer_lineage: Mapping[str, Any]


@dataclass(frozen=True)
class TomeArtifactValidation:
    producer_status: str
    producer_validated_by: str | None
    producer_report_path: str | None
    contract_status: str
    student_interpretation: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    unsupported_required_capabilities: tuple[str, ...]
    unknown_required_roles: tuple[str, ...]


@dataclass(frozen=True)
class TomeBehavioralSurface:
    surface_id: str
    surface_kind: str
    schema_version: str
    required_content_roles: tuple[str, ...]
    optional_content_roles: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    prerequisites: tuple[str, ...]
    target_scope: Mapping[str, Any]
    semantics: Mapping[str, Any]
    metadata: Mapping[str, Any]
    known_surface: bool


@dataclass(frozen=True)
class TomeCorridorView:
    surface_id: str
    mode_policy: str
    tracked_statistics: tuple[str, ...]
    mode_count: int
    assignment_count: int
    assignment_storage_kind: str
    corridor_stat_top_k: int
    degraded: bool
    required_capabilities: tuple[str, ...]
    content_references: tuple[ProductionTomeContentRef, ...]

    mode_identifier_type: ClassVar[type[ArtifactLocalModeId]] = ArtifactLocalModeId
    fingerprint_identifier_type: ClassVar[type[ArtifactLocalFingerprintId]] = (
        ArtifactLocalFingerprintId
    )


@dataclass(frozen=True)
class TomeExemplarView:
    surface_id: str
    selected_exemplar_count: int
    payload_shard_references: tuple[ProductionTomeContentRef, ...]
    dynamic_top_k_metadata: Mapping[str, Any]
    corridor_linkage_required: bool
    required_capabilities: tuple[str, ...]
    content_references: tuple[ProductionTomeContentRef, ...]


@dataclass(frozen=True)
class TomeArtifactView:
    artifact_dir: Path
    contract_family: Literal["production_v2", "legacy_dense_v0"]
    identity: TomeArtifactIdentity
    provenance: TomeArtifactProvenance
    validation: TomeArtifactValidation
    claims_not_made: tuple[str, ...]
    contents_index: tuple[ProductionTomeContentRef, ...]
    surfaces: tuple[TomeBehavioralSurface, ...]
    recommended_training_plan: RecommendedTrainingPlan | None
    warnings: tuple[str, ...]
    corridor_contract: TomeCorridorView | None = None
    exemplar_contract: TomeExemplarView | None = None

    # Legacy smoke/debug compatibility. These are never production sources of truth.
    cover_page: TomeCoverPage | None = None
    manifest: TomeManifest | None = None
    payload_summary: TomePayloadSummary | None = None
    vocab_contract: VocabContract | None = None
    tokenizer_contract: Mapping[str, Any] | None = None
    sequence_length: int | None = None
    record_count: int | None = None
    payload_format: TomePayloadFormat | None = None
    inferred_defaults: TomeInferredDefaults | None = None

    def surface(self, surface_id: str) -> TomeBehavioralSurface | None:
        return next(
            (surface for surface in self.surfaces if surface.surface_id == surface_id),
            None,
        )


class TomeArtifactError(ValueError):
    def __init__(self, path: str | Path, blockers: tuple[str, ...]) -> None:
        self.path = Path(path)
        self.blockers = blockers
        super().__init__(
            f"could not open Tome artifact at {self.path}: " + ", ".join(blockers)
        )


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
