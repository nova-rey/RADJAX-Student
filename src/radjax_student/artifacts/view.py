from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_contract.tome import (
    TomeCoverPage,
    TomeManifest,
    inspect_production_tome,
    load_production_tome,
    load_tome_cover_page,
    validate_tome,
)
from radjax_contract.tome.inspection import (
    TomeConsumptionPlan,
    inspect_tome_for_consumption,
)
from radjax_contract.tome.production import (
    BehavioralSurfaceContract,
    CorridorContract,
    ExemplarContract,
    ProductionTomeArtifact,
    ProductionTomeContentRef,
    ProductionTomeInspection,
    ProductionTomeValidationResult,
)
from radjax_contract.vocab import VocabContract

from radjax_student.artifacts.models import (
    TomeArtifactError,
    TomeArtifactIdentity,
    TomeArtifactProvenance,
    TomeArtifactValidation,
    TomeArtifactView,
    TomeBehavioralSurface,
    TomeCorridorView,
    TomeExemplarView,
    TomeInferredDefaults,
    TomePayloadSummary,
    freeze_mapping,
)


def open_tome_artifact(path: str | Path) -> TomeArtifactView:
    artifact_dir = Path(path)
    production = load_production_tome(artifact_dir)
    if production.artifact is not None:
        if production.blockers:
            raise TomeArtifactError(artifact_dir, production.blockers)
        inspection = inspect_production_tome(artifact_dir)
        if inspection.blockers:
            raise TomeArtifactError(artifact_dir, inspection.blockers)
        return _production_view(artifact_dir, production, inspection)
    return _open_legacy_dense_v0(artifact_dir, production.blockers)


def _production_view(
    artifact_dir: Path,
    validation: ProductionTomeValidationResult,
    inspection: ProductionTomeInspection,
) -> TomeArtifactView:
    artifact = validation.artifact
    if artifact is None:
        raise TomeArtifactError(artifact_dir, ("production_artifact_missing",))
    cover = artifact.cover_page
    identity = cover.identity
    warnings = _production_warnings(validation, inspection, cover.surfaces)
    surfaces = tuple(_surface_view(surface, inspection) for surface in cover.surfaces)
    targets = cover.provenance.targets
    return TomeArtifactView(
        artifact_dir=artifact_dir,
        contract_family="production_v2",
        identity=TomeArtifactIdentity(
            artifact_kind=identity.artifact_kind,
            cover_page_version=identity.cover_page_version,
            tome_version=identity.tome_version,
            layout=identity.layout,
            source_artifact_type=identity.source_artifact_type,
            created_by=identity.created_by,
            created_at=identity.created_at,
            producer_identity=identity.created_by,
        ),
        provenance=TomeArtifactProvenance(
            teacher=freeze_mapping(cover.provenance.teacher),
            tokenizer=freeze_mapping(cover.provenance.tokenizer),
            targets=freeze_mapping(targets),
            corpus=_optional_mapping(cover.provenance.corpus),
            teacher_model=_optional_mapping(cover.provenance.teacher_model),
            producer_lineage=freeze_mapping(
                {
                    "created_by": identity.created_by,
                    "created_at": identity.created_at,
                    "source_artifact_type": identity.source_artifact_type,
                    "validated_by": cover.producer_validation.validated_by,
                    "validation_report_path": (
                        cover.producer_validation.validation_report_path
                    ),
                }
            ),
        ),
        validation=TomeArtifactValidation(
            producer_status=cover.producer_validation.status,
            producer_validated_by=cover.producer_validation.validated_by,
            producer_report_path=cover.producer_validation.validation_report_path,
            contract_status=validation.status,
            student_interpretation=_student_interpretation(inspection),
            blockers=validation.blockers,
            warnings=validation.warnings,
            required_capabilities=inspection.required_capabilities,
            unsupported_required_capabilities=(
                inspection.unsupported_required_capabilities
            ),
            unknown_required_roles=inspection.unknown_required_roles,
        ),
        claims_not_made=cover.claims_not_made,
        contents_index=cover.contents,
        surfaces=surfaces,
        recommended_training_plan=cover.recommended_training_plan,
        warnings=warnings,
        corridor_contract=_corridor_view(artifact, cover.contents),
        exemplar_contract=_exemplar_view(artifact, cover.contents, targets),
        tokenizer_contract=freeze_mapping(cover.provenance.tokenizer),
        sequence_length=_optional_int(targets.get("sequence_length")),
        record_count=_optional_int(targets.get("num_examples")),
        inferred_defaults=_production_inferred_defaults(artifact),
    )


def _surface_view(
    surface: BehavioralSurfaceContract,
    inspection: ProductionTomeInspection,
) -> TomeBehavioralSurface:
    return TomeBehavioralSurface(
        surface_id=surface.surface_id,
        surface_kind=surface.surface_kind,
        schema_version=surface.schema_version,
        required_content_roles=surface.required_content_roles,
        optional_content_roles=surface.optional_content_roles,
        required_capabilities=surface.required_capabilities,
        prerequisites=surface.prerequisites,
        target_scope=freeze_mapping(surface.target_scope),
        semantics=freeze_mapping(surface.semantics),
        metadata=freeze_mapping(surface.metadata),
        known_surface=surface.surface_id in inspection.known_surfaces,
    )


def _corridor_view(
    artifact: ProductionTomeArtifact,
    contents: tuple[ProductionTomeContentRef, ...],
) -> TomeCorridorView | None:
    corridor: CorridorContract | None = artifact.corridor
    if corridor is None:
        return None
    surface = corridor.surface
    summary = corridor.summary
    return TomeCorridorView(
        surface_id=surface.surface_id,
        mode_policy=summary.mode_policy,
        tracked_statistics=summary.tracked_stats,
        mode_count=summary.mode_count,
        assignment_count=summary.assignment_count,
        assignment_storage_kind=summary.assignment_storage_kind,
        corridor_stat_top_k=summary.corridor_stat_top_k,
        degraded=summary.degraded,
        required_capabilities=surface.required_capabilities,
        content_references=_surface_content_references(surface, contents),
    )


def _exemplar_view(
    artifact: ProductionTomeArtifact,
    contents: tuple[ProductionTomeContentRef, ...],
    targets: dict[str, Any],
) -> TomeExemplarView | None:
    exemplar: ExemplarContract | None = artifact.exemplar
    if exemplar is None:
        return None
    surface = exemplar.surface
    target_params = targets.get("target_params", {})
    if not isinstance(target_params, Mapping):
        target_params = {}
    dynamic_metadata = {
        str(key): value
        for key, value in target_params.items()
        if str(key).startswith("dynamic_")
    }
    return TomeExemplarView(
        surface_id=surface.surface_id,
        selected_exemplar_count=len(exemplar.selected_index),
        payload_shard_references=tuple(
            ref for ref in contents if ref.role == "selected_exemplar_payload_shard"
        ),
        dynamic_top_k_metadata=freeze_mapping(dynamic_metadata),
        corridor_linkage_required="corridor" in surface.prerequisites,
        required_capabilities=surface.required_capabilities,
        content_references=_surface_content_references(surface, contents),
    )


def _surface_content_references(
    surface: BehavioralSurfaceContract,
    contents: tuple[ProductionTomeContentRef, ...],
) -> tuple[ProductionTomeContentRef, ...]:
    roles = set(surface.required_content_roles) | set(surface.optional_content_roles)
    return tuple(ref for ref in contents if ref.role in roles)


def _production_warnings(
    validation: ProductionTomeValidationResult,
    inspection: ProductionTomeInspection,
    surfaces: tuple[BehavioralSurfaceContract, ...],
) -> tuple[str, ...]:
    warnings = list(validation.warnings)
    warnings.extend(
        f"unsupported_required_capability: {capability}"
        for capability in inspection.unsupported_required_capabilities
    )
    warnings.extend(
        f"unknown_required_role: {role}" for role in inspection.unknown_required_roles
    )
    surface_by_id = {surface.surface_id: surface for surface in surfaces}
    warnings.extend(
        f"unknown_surface_kind: {surface_id}={surface_by_id[surface_id].surface_kind}"
        for surface_id in inspection.unknown_surfaces
    )
    return tuple(dict.fromkeys(warnings))


def _student_interpretation(inspection: ProductionTomeInspection) -> str:
    if (
        inspection.unsupported_required_capabilities
        or inspection.unknown_required_roles
    ):
        return "metadata_only_requires_capabilities"
    return "metadata_only_ready"


def _production_inferred_defaults(
    artifact: ProductionTomeArtifact,
) -> TomeInferredDefaults:
    provenance = artifact.cover_page.provenance
    teacher = provenance.teacher
    tokenizer = provenance.tokenizer
    return TomeInferredDefaults(
        role=None,
        teacher_id=_optional_string(teacher.get("model_id")),
        teacher_family=_optional_string(teacher.get("model_family")),
        teacher_backend=_optional_string(teacher.get("backend_type")),
        tokenizer_id=_optional_string(tokenizer.get("tokenizer_id")),
        vocab_size=_optional_int(tokenizer.get("vocab_size")),
        adapter_family=None,
        compression_family=None,
        requires_reconstruction=None,
    )


def _open_legacy_dense_v0(
    artifact_dir: Path,
    production_blockers: tuple[str, ...],
) -> TomeArtifactView:
    validation = validate_tome(artifact_dir)
    cover_result = load_tome_cover_page(artifact_dir)
    consumption_plan = inspect_tome_for_consumption(artifact_dir)
    legacy_blockers = _collect_blockers(
        validation_blockers=validation.blockers,
        cover_blockers=cover_result.blockers,
        consumption_blockers=consumption_plan.blockers,
    )
    if legacy_blockers:
        raise TomeArtifactError(
            artifact_dir,
            _collect_blockers(
                validation_blockers=production_blockers,
                cover_blockers=legacy_blockers,
                consumption_blockers=(),
            ),
        )
    if validation.manifest is None:
        raise TomeArtifactError(artifact_dir, ("manifest_missing",))
    if cover_result.cover_page is None:
        raise TomeArtifactError(artifact_dir, ("cover_page_missing",))
    return _legacy_view(
        artifact_dir,
        validation.manifest,
        cover_result.cover_page,
        consumption_plan,
        tuple(validation.warnings),
    )


def _legacy_view(
    artifact_dir: Path,
    manifest: TomeManifest,
    cover_page: TomeCoverPage,
    consumption_plan: TomeConsumptionPlan,
    contract_warnings: tuple[str, ...],
) -> TomeArtifactView:
    warnings = tuple(dict.fromkeys((*contract_warnings, "legacy_dense_v0_smoke_path")))
    vocab = manifest.vocab_contract
    tokenizer = _tokenizer_contract(vocab)
    return TomeArtifactView(
        artifact_dir=artifact_dir,
        contract_family="legacy_dense_v0",
        identity=TomeArtifactIdentity(
            artifact_kind=manifest.artifact_kind,
            cover_page_version=cover_page.cover_page_version,
            tome_version=None,
            layout="legacy_dense_directory",
            source_artifact_type="dense_logits_smoke",
            created_by=None,
            created_at=None,
            producer_identity=manifest.producer,
        ),
        provenance=TomeArtifactProvenance(
            teacher=freeze_mapping(
                {
                    "teacher_id": cover_page.teacher.teacher_id,
                    "teacher_family": cover_page.teacher.teacher_family,
                    "backend": cover_page.teacher.backend,
                }
            ),
            tokenizer=freeze_mapping({} if tokenizer is None else tokenizer),
            targets=freeze_mapping(
                {
                    "record_count": manifest.record_count,
                    "sequence_length": manifest.sequence_length,
                    "payload_format": manifest.payload_format.value,
                }
            ),
            corpus=freeze_mapping(
                {
                    "summary": cover_page.corpus.summary,
                    "contains_synthetic_examples": (
                        cover_page.corpus.contains_synthetic_examples
                    ),
                }
            ),
            teacher_model=None,
            producer_lineage=freeze_mapping(
                {
                    "producer": manifest.producer,
                    "schema_version": manifest.schema_version,
                }
            ),
        ),
        validation=TomeArtifactValidation(
            producer_status="not_reported",
            producer_validated_by=None,
            producer_report_path=None,
            contract_status="pass",
            student_interpretation="legacy_smoke_debug_only",
            blockers=(),
            warnings=contract_warnings,
            required_capabilities=(),
            unsupported_required_capabilities=(),
            unknown_required_roles=(),
        ),
        claims_not_made=(),
        contents_index=(),
        surfaces=(),
        recommended_training_plan=None,
        warnings=warnings,
        cover_page=cover_page,
        manifest=manifest,
        payload_summary=_payload_summary(manifest, cover_page, consumption_plan),
        vocab_contract=vocab,
        tokenizer_contract=None if tokenizer is None else freeze_mapping(tokenizer),
        sequence_length=manifest.sequence_length,
        record_count=manifest.record_count,
        payload_format=manifest.payload_format,
        inferred_defaults=_legacy_inferred_defaults(
            manifest, cover_page, consumption_plan
        ),
    )


def _collect_blockers(
    *,
    validation_blockers: tuple[str, ...],
    cover_blockers: tuple[str, ...],
    consumption_blockers: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*validation_blockers, *cover_blockers, *consumption_blockers))
    )


def _payload_summary(
    manifest: TomeManifest,
    cover_page: TomeCoverPage,
    consumption_plan: TomeConsumptionPlan,
) -> TomePayloadSummary:
    return TomePayloadSummary(
        payload_format=manifest.payload_format,
        compression=manifest.compression,
        expected_adapter=consumption_plan.adapter_id
        or cover_page.student_consumption.expected_adapter,
        implemented_by_contract=cover_page.student_consumption.implemented_by_contract,
        record_count=manifest.record_count,
        sequence_length=manifest.sequence_length,
        shard_count=manifest.shard_count,
        shard_paths=tuple(shard.path for shard in manifest.shards),
    )


def _tokenizer_contract(vocab_contract: VocabContract | None) -> dict[str, Any] | None:
    if vocab_contract is None:
        return None
    return {
        "tokenizer_id": vocab_contract.tokenizer_id,
        "tokenizer_hash": vocab_contract.tokenizer_hash,
        "model_id": vocab_contract.model_id,
        "model_family": vocab_contract.model_family,
        "special_tokens": dict(vocab_contract.special_tokens),
    }


def _legacy_inferred_defaults(
    manifest: TomeManifest,
    cover_page: TomeCoverPage,
    consumption_plan: TomeConsumptionPlan,
) -> TomeInferredDefaults:
    vocab = manifest.vocab_contract
    return TomeInferredDefaults(
        role=manifest.role.value,
        teacher_id=cover_page.teacher.teacher_id,
        teacher_family=cover_page.teacher.teacher_family,
        teacher_backend=cover_page.teacher.backend,
        tokenizer_id=None if vocab is None else vocab.tokenizer_id,
        vocab_size=None if vocab is None else vocab.vocab_size,
        adapter_family=consumption_plan.adapter_id
        or cover_page.student_consumption.expected_adapter,
        compression_family=manifest.compression.family.value,
        requires_reconstruction=manifest.compression.requires_reconstruction,
    )


def _optional_mapping(value: dict[str, Any] | None) -> Mapping[str, Any] | None:
    return None if value is None else freeze_mapping(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
