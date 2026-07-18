from __future__ import annotations

from pathlib import Path
from typing import Any

from radjax_student.artifacts.default_models import (
    ArtifactRunFacts,
    AvailableSurface,
    CorridorSurfaceFacts,
    ExemplarSurfaceFacts,
    RecommendedPass,
    StudentRunDefaults,
    immutable_mapping,
)
from radjax_student.artifacts.models import (
    TomeArtifactView,
    TomeBehavioralSurface,
)
from radjax_student.artifacts.view import open_tome_artifact

REQUIRED_FROM_USER: dict[str, None] = {
    "student_architecture": None,
    "student_size_or_config": None,
    "training_budget": None,
    "output_dir": None,
}

UNRESOLVED_BY_PHASE: dict[str, str] = {
    "runtime_backend": "phase_2_or_later",
    "precision_policy": "phase_2_or_later",
    "optimizer": "phase_3_or_later",
    "schedule_implementation": "phase_3_or_later",
    "loss_weighting": "phase_3_or_later",
    "architecture_plugin": "phase_4",
    "evaluation_policy": "phase_5_or_later",
    "hf_export_details": "phase_5_or_later",
}

STUDENT_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_not_run",
    "compatibility_not_evaluated",
    "required_capabilities_not_proven",
    "student_architecture_not_selected",
    "runtime_not_selected",
    "optimizer_not_selected",
    "schedule_not_executable",
    "payload_tensors_not_loaded",
    "model_not_allocated",
    "model_quality_not_claimed",
    "hf_export_not_ready",
    "radlads_parity_not_measured",
)


def infer_run_defaults(view: TomeArtifactView) -> StudentRunDefaults:
    artifact_facts = _artifact_facts(view)
    if view.contract_family == "legacy_dense_v0":
        return _legacy_defaults(view, artifact_facts)

    surfaces = tuple(_surface_defaults(view, surface) for surface in view.surfaces)
    required_capabilities = _required_capabilities(view, surfaces)
    plan = _recommended_plan(view)
    return StudentRunDefaults(
        artifact_facts=artifact_facts,
        available_surfaces=surfaces,
        required_capabilities=required_capabilities,
        unsupported_required_capabilities=tuple(
            sorted(set(view.validation.unsupported_required_capabilities))
        ),
        capabilities_not_yet_evaluated=required_capabilities,
        recommended_training_plan=plan,
        required_from_user=immutable_mapping(REQUIRED_FROM_USER),
        unresolved_by_phase=immutable_mapping(UNRESOLVED_BY_PHASE),
        warnings=tuple(view.warnings),
        artifact_claims_not_made=view.claims_not_made,
        student_claims_not_made=STUDENT_CLAIMS_NOT_MADE,
    )


def infer_run_defaults_from_tome(path: str | Path) -> StudentRunDefaults:
    return infer_run_defaults(open_tome_artifact(path))


def _artifact_facts(view: TomeArtifactView) -> ArtifactRunFacts:
    teacher = view.provenance.teacher
    tokenizer = view.provenance.tokenizer
    teacher_model = view.provenance.teacher_model or {}
    return ArtifactRunFacts(
        contract_family=view.contract_family,
        artifact_kind=view.identity.artifact_kind,
        cover_page_version=view.identity.cover_page_version,
        tome_version=view.identity.tome_version,
        layout=view.identity.layout,
        source_artifact_type=view.identity.source_artifact_type,
        artifact_id=None,
        producer_identity=view.identity.producer_identity,
        teacher_model_identity=_optional_string(
            teacher.get("model_id", teacher.get("teacher_id"))
        ),
        teacher_model_revision=_optional_string(teacher_model.get("model_revision")),
        teacher_family=_optional_string(
            teacher.get("model_family", teacher.get("teacher_family"))
        ),
        teacher_backend=_optional_string(
            teacher.get("backend_type", teacher.get("backend"))
        ),
        tokenizer_id=_optional_string(tokenizer.get("tokenizer_id")),
        tokenizer_hash=_optional_string(tokenizer.get("tokenizer_hash")),
        vocab_size=_optional_int(tokenizer.get("vocab_size")),
        sequence_length=view.sequence_length,
        example_count=view.record_count,
        producer_validation_status=view.validation.producer_status,
        contract_validation_status=view.validation.contract_status,
        content_count=len(view.contents_index),
        surface_count=len(view.surfaces),
    )


def _surface_defaults(
    view: TomeArtifactView,
    surface: TomeBehavioralSurface,
) -> AvailableSurface:
    corridor = view.corridor_contract
    exemplar = view.exemplar_contract
    return AvailableSurface(
        surface_id=surface.surface_id,
        surface_kind=surface.surface_kind,
        schema_version=surface.schema_version,
        known_surface=surface.known_surface,
        required_capabilities=surface.required_capabilities,
        prerequisites=surface.prerequisites,
        target_scope=immutable_mapping(surface.target_scope),
        semantics=immutable_mapping(surface.semantics),
        required_content_roles=surface.required_content_roles,
        optional_content_roles=surface.optional_content_roles,
        corridor=(
            None
            if corridor is None or corridor.surface_id != surface.surface_id
            else CorridorSurfaceFacts(
                mode_policy=corridor.mode_policy,
                tracked_statistics=corridor.tracked_statistics,
                mode_count=corridor.mode_count,
                assignment_count=corridor.assignment_count,
                assignment_storage_kind=corridor.assignment_storage_kind,
                corridor_stat_top_k=corridor.corridor_stat_top_k,
                degraded=corridor.degraded,
            )
        ),
        exemplar=(
            None
            if exemplar is None or exemplar.surface_id != surface.surface_id
            else ExemplarSurfaceFacts(
                selected_exemplar_count=exemplar.selected_exemplar_count,
                dynamic_top_k_metadata=immutable_mapping(
                    exemplar.dynamic_top_k_metadata
                ),
                payload_shard_count=len(exemplar.payload_shard_references),
                corridor_linkage_required=exemplar.corridor_linkage_required,
            )
        ),
    )


def _required_capabilities(
    view: TomeArtifactView,
    surfaces: tuple[AvailableSurface, ...],
) -> tuple[str, ...]:
    capabilities = set(view.validation.required_capabilities)
    for surface in surfaces:
        capabilities.update(surface.required_capabilities)
    plan = view.recommended_training_plan
    if plan is not None:
        for training_pass in plan.passes:
            capabilities.update(training_pass.required_capabilities)
    return tuple(sorted(capabilities))


def _recommended_plan(view: TomeArtifactView) -> tuple[RecommendedPass, ...]:
    plan = view.recommended_training_plan
    if plan is None:
        return ()
    surface_by_id = {surface.surface_id: surface for surface in view.surfaces}
    return tuple(
        RecommendedPass(
            pass_index=index,
            pass_id=training_pass.pass_id,
            surface_id=training_pass.surface_id,
            checkpoint_after=training_pass.checkpoint_after,
            prerequisites=training_pass.prerequisites,
            required_capabilities=training_pass.required_capabilities,
            target_scope=immutable_mapping(
                surface_by_id[training_pass.surface_id].target_scope
            ),
        )
        for index, training_pass in enumerate(plan.passes)
    )


def _legacy_defaults(
    view: TomeArtifactView,
    artifact_facts: ArtifactRunFacts,
) -> StudentRunDefaults:
    defaults = view.inferred_defaults
    if defaults is None:
        raise ValueError("legacy Tome artifact view does not expose smoke defaults")
    legacy = immutable_mapping(
        {
            "artifact_role": defaults.role,
            "teacher_id": defaults.teacher_id,
            "teacher_family": defaults.teacher_family,
            "teacher_backend": defaults.teacher_backend,
            "tokenizer_id": defaults.tokenizer_id,
            "vocab_size": defaults.vocab_size,
            "sequence_length": view.sequence_length,
            "record_count": view.record_count,
            "payload_format": (
                None if view.payload_format is None else view.payload_format.value
            ),
            "compression_family": defaults.compression_family,
            "requires_reconstruction": defaults.requires_reconstruction,
            "expected_adapter_family": defaults.adapter_family,
            "classification": "legacy_dense_v0_smoke_only",
        }
    )
    return StudentRunDefaults(
        artifact_facts=artifact_facts,
        available_surfaces=(),
        required_capabilities=(),
        unsupported_required_capabilities=(),
        capabilities_not_yet_evaluated=(),
        recommended_training_plan=(),
        required_from_user=immutable_mapping(REQUIRED_FROM_USER),
        unresolved_by_phase=immutable_mapping(UNRESOLVED_BY_PHASE),
        warnings=tuple(view.warnings),
        artifact_claims_not_made=view.claims_not_made,
        student_claims_not_made=STUDENT_CLAIMS_NOT_MADE,
        legacy_smoke_defaults=legacy,
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
