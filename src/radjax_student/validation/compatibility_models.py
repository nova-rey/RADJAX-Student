from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from radjax_student.validation.default_models import immutable_mapping

CompatibilityStatus = Literal["pass", "fail"]
CheckStatus = Literal["pass", "fail", "unevaluated", "not_present"]


@dataclass(frozen=True)
class StudentCapabilityProfile:
    profile_id: str
    supported_contract_families: tuple[str, ...]
    supported_tome_versions: tuple[int, ...]
    supported_cover_page_versions: tuple[int | str, ...]
    supported_surface_kinds: tuple[str, ...]
    supported_surface_schemas: tuple[tuple[str, str], ...]
    supported_capabilities: tuple[str, ...]
    supported_target_scopes: tuple[str, ...]
    max_sequence_length: int | None
    max_vocab_size: int | None
    supported_tokenizer_ids: tuple[str, ...] | None = None
    supports_ordered_passes: bool = True
    supports_checkpoint_boundaries: bool = True
    supports_pass_prerequisites: bool = True
    accepted_producer_statuses: tuple[str, ...] = ("pass",)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be nonempty")
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive when declared")
        if self.max_vocab_size is not None and self.max_vocab_size <= 0:
            raise ValueError("max_vocab_size must be positive when declared")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "supported_contract_families": list(self.supported_contract_families),
            "supported_tome_versions": list(self.supported_tome_versions),
            "supported_cover_page_versions": list(self.supported_cover_page_versions),
            "supported_surface_kinds": list(self.supported_surface_kinds),
            "supported_surface_schemas": [
                {"surface_kind": kind, "schema_version": schema}
                for kind, schema in self.supported_surface_schemas
            ],
            "supported_capabilities": list(self.supported_capabilities),
            "supported_target_scopes": list(self.supported_target_scopes),
            "max_sequence_length": self.max_sequence_length,
            "max_vocab_size": self.max_vocab_size,
            "supported_tokenizer_ids": (
                None
                if self.supported_tokenizer_ids is None
                else list(self.supported_tokenizer_ids)
            ),
            "supports_ordered_passes": self.supports_ordered_passes,
            "supports_checkpoint_boundaries": self.supports_checkpoint_boundaries,
            "supports_pass_prerequisites": self.supports_pass_prerequisites,
            "accepted_producer_statuses": list(self.accepted_producer_statuses),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CompatibilityFinding:
    code: str
    message: str
    details: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        code: str,
        message: str,
        **details: Any,
    ) -> CompatibilityFinding:
        return cls(code=code, message=message, details=immutable_mapping(details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": _json_value(self.details),
        }


@dataclass(frozen=True)
class DimensionCompatibility:
    status: CheckStatus
    artifact_value: int | None
    profile_limit: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact_value": self.artifact_value,
            "profile_limit": self.profile_limit,
        }


@dataclass(frozen=True)
class TargetScopeCompatibility:
    surface_id: str
    scope_kind: str
    required: bool
    status: CheckStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "scope_kind": self.scope_kind,
            "required": self.required,
            "status": self.status,
        }


@dataclass(frozen=True)
class PlanCompatibility:
    status: CheckStatus
    pass_count: int
    ordered_passes_supported: bool
    checkpoint_boundaries_supported: bool
    prerequisites_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pass_count": self.pass_count,
            "ordered_passes_supported": self.ordered_passes_supported,
            "checkpoint_boundaries_supported": (self.checkpoint_boundaries_supported),
            "prerequisites_supported": self.prerequisites_supported,
        }


@dataclass(frozen=True)
class CompatibilityArtifactIdentity:
    contract_family: str
    artifact_kind: str
    cover_page_version: int | str
    tome_version: int | None
    source_artifact_type: str
    producer_validation_status: str
    contract_validation_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_family": self.contract_family,
            "artifact_kind": self.artifact_kind,
            "cover_page_version": self.cover_page_version,
            "tome_version": self.tome_version,
            "source_artifact_type": self.source_artifact_type,
            "producer_validation_status": self.producer_validation_status,
            "contract_validation_status": self.contract_validation_status,
        }


@dataclass(frozen=True)
class StudentCompatibilityReport:
    status: CompatibilityStatus
    profile_id: str
    artifact_identity: CompatibilityArtifactIdentity
    required_capabilities: tuple[str, ...]
    supported_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    unevaluated_capabilities: tuple[str, ...]
    supported_surfaces: tuple[str, ...]
    unsupported_surfaces: tuple[str, ...]
    sequence_compatibility: DimensionCompatibility
    vocab_compatibility: DimensionCompatibility
    target_scope_compatibility: tuple[TargetScopeCompatibility, ...]
    plan_compatibility: PlanCompatibility
    blockers: tuple[CompatibilityFinding, ...]
    warnings: tuple[CompatibilityFinding, ...]
    artifact_claims_not_made: tuple[str, ...]
    student_claims_not_made: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "profile_id": self.profile_id,
            "artifact_identity": self.artifact_identity.to_dict(),
            "required_capabilities": list(self.required_capabilities),
            "supported_capabilities": list(self.supported_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "unevaluated_capabilities": list(self.unevaluated_capabilities),
            "supported_surfaces": list(self.supported_surfaces),
            "unsupported_surfaces": list(self.unsupported_surfaces),
            "sequence_compatibility": self.sequence_compatibility.to_dict(),
            "vocab_compatibility": self.vocab_compatibility.to_dict(),
            "target_scope_compatibility": [
                item.to_dict() for item in self.target_scope_compatibility
            ],
            "plan_compatibility": self.plan_compatibility.to_dict(),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "artifact_claims_not_made": list(self.artifact_claims_not_made),
            "student_claims_not_made": list(self.student_claims_not_made),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
