from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ArtifactRunFacts:
    contract_family: str
    artifact_kind: str
    cover_page_version: int | str
    tome_version: int | None
    layout: str
    source_artifact_type: str
    artifact_id: str | None
    producer_identity: str
    teacher_model_identity: str | None
    teacher_model_revision: str | None
    teacher_family: str | None
    teacher_backend: str | None
    tokenizer_id: str | None
    tokenizer_hash: str | None
    vocab_size: int | None
    sequence_length: int | None
    example_count: int | None
    producer_validation_status: str
    contract_validation_status: str
    content_count: int
    surface_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_family": self.contract_family,
            "artifact_kind": self.artifact_kind,
            "cover_page_version": self.cover_page_version,
            "tome_version": self.tome_version,
            "layout": self.layout,
            "source_artifact_type": self.source_artifact_type,
            "artifact_id": self.artifact_id,
            "producer_identity": self.producer_identity,
            "teacher_model_identity": self.teacher_model_identity,
            "teacher_model_revision": self.teacher_model_revision,
            "teacher_family": self.teacher_family,
            "teacher_backend": self.teacher_backend,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_hash": self.tokenizer_hash,
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "example_count": self.example_count,
            "producer_validation_status": self.producer_validation_status,
            "contract_validation_status": self.contract_validation_status,
            "content_count": self.content_count,
            "surface_count": self.surface_count,
        }


@dataclass(frozen=True)
class CorridorSurfaceFacts:
    mode_policy: str
    tracked_statistics: tuple[str, ...]
    mode_count: int
    assignment_count: int
    assignment_storage_kind: str
    corridor_stat_top_k: int
    degraded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_policy": self.mode_policy,
            "tracked_statistics": list(self.tracked_statistics),
            "mode_count": self.mode_count,
            "assignment_count": self.assignment_count,
            "assignment_storage_kind": self.assignment_storage_kind,
            "corridor_stat_top_k": self.corridor_stat_top_k,
            "degraded": self.degraded,
        }


@dataclass(frozen=True)
class ExemplarSurfaceFacts:
    selected_exemplar_count: int
    dynamic_top_k_metadata: Mapping[str, Any]
    payload_shard_count: int
    corridor_linkage_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_exemplar_count": self.selected_exemplar_count,
            "dynamic_top_k_metadata": _json_value(self.dynamic_top_k_metadata),
            "payload_shard_count": self.payload_shard_count,
            "corridor_linkage_required": self.corridor_linkage_required,
        }


@dataclass(frozen=True)
class AvailableSurface:
    surface_id: str
    surface_kind: str
    schema_version: str
    known_surface: bool
    required_capabilities: tuple[str, ...]
    prerequisites: tuple[str, ...]
    target_scope: Mapping[str, Any]
    semantics: Mapping[str, Any]
    required_content_roles: tuple[str, ...]
    optional_content_roles: tuple[str, ...]
    corridor: CorridorSurfaceFacts | None = None
    exemplar: ExemplarSurfaceFacts | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "surface_kind": self.surface_kind,
            "schema_version": self.schema_version,
            "known_surface": self.known_surface,
            "required_capabilities": list(self.required_capabilities),
            "prerequisites": list(self.prerequisites),
            "target_scope": _json_value(self.target_scope),
            "semantics": _json_value(self.semantics),
            "required_content_roles": list(self.required_content_roles),
            "optional_content_roles": list(self.optional_content_roles),
            "corridor": None if self.corridor is None else self.corridor.to_dict(),
            "exemplar": None if self.exemplar is None else self.exemplar.to_dict(),
        }


@dataclass(frozen=True)
class RecommendedPass:
    pass_index: int
    pass_id: str
    surface_id: str
    checkpoint_after: bool
    prerequisites: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    target_scope: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "pass_id": self.pass_id,
            "surface_id": self.surface_id,
            "checkpoint_after": self.checkpoint_after,
            "prerequisites": list(self.prerequisites),
            "required_capabilities": list(self.required_capabilities),
            "target_scope": _json_value(self.target_scope),
        }


@dataclass(frozen=True)
class StudentRunDefaults:
    artifact_facts: ArtifactRunFacts
    available_surfaces: tuple[AvailableSurface, ...]
    required_capabilities: tuple[str, ...]
    unsupported_required_capabilities: tuple[str, ...]
    capabilities_not_yet_evaluated: tuple[str, ...]
    recommended_training_plan: tuple[RecommendedPass, ...]
    required_from_user: Mapping[str, None]
    unresolved_by_phase: Mapping[str, str]
    warnings: tuple[str, ...]
    artifact_claims_not_made: tuple[str, ...]
    student_claims_not_made: tuple[str, ...]
    legacy_smoke_defaults: Mapping[str, Any] | None = None

    @property
    def inferred_from_tome(self) -> dict[str, Any]:
        """Deprecated compatibility view for legacy dense smoke callers only."""

        return (
            {}
            if self.legacy_smoke_defaults is None
            else dict(self.legacy_smoke_defaults)
        )

    @property
    def claims_not_made(self) -> tuple[str, ...]:
        """Deprecated alias for pre-P1.7 Student-local non-claims."""

        return self.student_claims_not_made

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_facts": self.artifact_facts.to_dict(),
            "available_surfaces": [
                surface.to_dict() for surface in self.available_surfaces
            ],
            "required_capabilities": list(self.required_capabilities),
            "unsupported_required_capabilities": list(
                self.unsupported_required_capabilities
            ),
            "capabilities_not_yet_evaluated": list(self.capabilities_not_yet_evaluated),
            "recommended_training_plan": [
                training_pass.to_dict()
                for training_pass in self.recommended_training_plan
            ],
            "required_from_user": dict(self.required_from_user),
            "unresolved_by_phase": dict(self.unresolved_by_phase),
            "warnings": list(self.warnings),
            "artifact_claims_not_made": list(self.artifact_claims_not_made),
            "student_claims_not_made": list(self.student_claims_not_made),
            "legacy_smoke_defaults": (
                None
                if self.legacy_smoke_defaults is None
                else _json_value(self.legacy_smoke_defaults)
            ),
        }


def immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return immutable_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
