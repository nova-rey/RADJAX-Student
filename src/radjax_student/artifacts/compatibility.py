from __future__ import annotations

from pathlib import Path

from radjax_student.artifacts.compatibility_models import (
    CompatibilityArtifactIdentity,
    CompatibilityFinding,
    DimensionCompatibility,
    PlanCompatibility,
    StudentCapabilityProfile,
    StudentCompatibilityReport,
    TargetScopeCompatibility,
)
from radjax_student.artifacts.default_models import (
    AvailableSurface,
    StudentRunDefaults,
)
from radjax_student.artifacts.models import TomeArtifactView
from radjax_student.artifacts.run_defaults import infer_run_defaults
from radjax_student.artifacts.view import open_tome_artifact

REPORT_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "payload_loading_not_tested",
    "loss_computation_not_tested",
    "training_not_run",
    "checkpoint_execution_not_tested",
    "runtime_not_selected",
    "architecture_not_instantiated",
    "capability_declaration_not_implementation_proof",
    "model_quality_not_claimed",
    "hf_export_not_ready",
    "radlads_parity_not_measured",
)


def metadata_inspection_only_profile() -> StudentCapabilityProfile:
    return StudentCapabilityProfile(
        profile_id="metadata_inspection_only",
        supported_contract_families=("production_v2",),
        supported_tome_versions=(1,),
        supported_cover_page_versions=(2,),
        supported_surface_kinds=("fingerprint_corridor", "selected_exemplar"),
        supported_surface_schemas=(
            ("fingerprint_corridor", "behavioral_surface_v1"),
            ("selected_exemplar", "behavioral_surface_v1"),
        ),
        supported_capabilities=(),
        supported_target_scopes=("whole_model", "unspecified", "default"),
        max_sequence_length=None,
        max_vocab_size=None,
        supported_tokenizer_ids=None,
        notes=(
            "metadata inspection only",
            "does not declare corridor or exemplar payload consumption",
        ),
    )


def evaluate_student_compatibility(
    view: TomeArtifactView,
    defaults: StudentRunDefaults,
    capability_profile: StudentCapabilityProfile,
) -> StudentCompatibilityReport:
    blockers: list[CompatibilityFinding] = []
    warnings: list[CompatibilityFinding] = []
    _check_identity(view, defaults, capability_profile, blockers)
    required = tuple(sorted(set(defaults.required_capabilities)))
    profile_capabilities = set(capability_profile.supported_capabilities)
    supported = tuple(item for item in required if item in profile_capabilities)
    missing = tuple(item for item in required if item not in profile_capabilities)
    for capability in missing:
        blockers.append(
            CompatibilityFinding.create(
                "missing_required_capability",
                f"required capability is not declared by profile: {capability}",
                capability=capability,
            )
        )
    if supported:
        warnings.append(
            CompatibilityFinding.create(
                "capability_declaration_not_execution_proof",
                "declared capability support has not executed payload behavior",
                capabilities=supported,
            )
        )

    required_surface_ids = {
        training_pass.surface_id for training_pass in defaults.recommended_training_plan
    }
    supported_surfaces, unsupported_surfaces, scopes = _check_surfaces(
        defaults.available_surfaces,
        required_surface_ids,
        capability_profile,
        blockers,
        warnings,
    )
    sequence = _check_dimension(
        dimension="sequence_length",
        artifact_value=defaults.artifact_facts.sequence_length,
        profile_limit=capability_profile.max_sequence_length,
        exceeds_code="sequence_length_exceeds_profile",
        unevaluated_code="sequence_compatibility_unevaluated",
        blockers=blockers,
    )
    vocab = _check_dimension(
        dimension="vocab_size",
        artifact_value=defaults.artifact_facts.vocab_size,
        profile_limit=capability_profile.max_vocab_size,
        exceeds_code="vocab_size_exceeds_profile",
        unevaluated_code="vocab_compatibility_unevaluated",
        blockers=blockers,
    )
    _check_tokenizer(defaults, capability_profile, blockers, warnings)
    plan = _check_plan(defaults, capability_profile, blockers)
    warnings.extend(
        CompatibilityFinding.create(
            "defaults_warning",
            warning,
            source="StudentRunDefaults",
        )
        for warning in defaults.warnings
        if not _resolved_capability_warning(warning, profile_capabilities)
    )
    warnings.append(
        CompatibilityFinding.create(
            "architecture_specific_compatibility_not_evaluated",
            "no architecture plugin or model configuration was instantiated",
        )
    )
    warnings.append(
        CompatibilityFinding.create(
            "special_token_compatibility_not_evaluated",
            "special-token binding requires a future architecture configuration",
        )
    )
    return StudentCompatibilityReport(
        status="pass" if not blockers else "fail",
        profile_id=capability_profile.profile_id,
        artifact_identity=_artifact_identity(defaults),
        required_capabilities=required,
        supported_capabilities=supported,
        missing_capabilities=missing,
        unevaluated_capabilities=supported,
        supported_surfaces=supported_surfaces,
        unsupported_surfaces=unsupported_surfaces,
        sequence_compatibility=sequence,
        vocab_compatibility=vocab,
        target_scope_compatibility=scopes,
        plan_compatibility=plan,
        blockers=tuple(blockers),
        warnings=_deduplicate_findings(warnings),
        artifact_claims_not_made=defaults.artifact_claims_not_made,
        student_claims_not_made=REPORT_CLAIMS_NOT_MADE,
    )


def evaluate_tome_path_compatibility(
    path: str | Path,
    capability_profile: StudentCapabilityProfile,
) -> StudentCompatibilityReport:
    view = open_tome_artifact(path)
    return evaluate_student_compatibility(
        view,
        infer_run_defaults(view),
        capability_profile,
    )


def _check_identity(
    view: TomeArtifactView,
    defaults: StudentRunDefaults,
    profile: StudentCapabilityProfile,
    blockers: list[CompatibilityFinding],
) -> None:
    facts = defaults.artifact_facts
    if facts.contract_family != view.contract_family:
        blockers.append(
            CompatibilityFinding.create(
                "defaults_artifact_mismatch",
                "run defaults and artifact view identify different contract families",
                view_contract_family=view.contract_family,
                defaults_contract_family=facts.contract_family,
            )
        )
    if facts.contract_validation_status != "pass":
        blockers.append(
            CompatibilityFinding.create(
                "contract_validation_failed",
                "Contract validation status is not pass",
                status=facts.contract_validation_status,
            )
        )
    if facts.producer_validation_status not in profile.accepted_producer_statuses:
        blockers.append(
            CompatibilityFinding.create(
                "producer_validation_failed",
                "producer validation status is not accepted by the profile",
                status=facts.producer_validation_status,
            )
        )
    if facts.contract_family not in profile.supported_contract_families:
        blockers.append(
            CompatibilityFinding.create(
                "unsupported_contract_family",
                "artifact contract family is not supported by the profile",
                contract_family=facts.contract_family,
            )
        )
    if (
        facts.tome_version is None
        or facts.tome_version not in profile.supported_tome_versions
    ):
        blockers.append(
            CompatibilityFinding.create(
                "unsupported_tome_version",
                "artifact Tome version is not supported by the profile",
                tome_version=facts.tome_version,
            )
        )
    if facts.cover_page_version not in profile.supported_cover_page_versions:
        blockers.append(
            CompatibilityFinding.create(
                "unsupported_cover_page_version",
                "cover-page version is not supported by the profile",
                cover_page_version=facts.cover_page_version,
            )
        )


def _check_surfaces(
    surfaces: tuple[AvailableSurface, ...],
    required_surface_ids: set[str],
    profile: StudentCapabilityProfile,
    blockers: list[CompatibilityFinding],
    warnings: list[CompatibilityFinding],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[TargetScopeCompatibility, ...],
]:
    supported: list[str] = []
    unsupported: list[str] = []
    scopes: list[TargetScopeCompatibility] = []
    supported_kinds = set(profile.supported_surface_kinds)
    supported_schemas = set(profile.supported_surface_schemas)
    supported_scopes = set(profile.supported_target_scopes)
    for surface in surfaces:
        required = surface.surface_id in required_surface_ids
        if not surface.known_surface:
            unsupported.append(surface.surface_id)
            finding = CompatibilityFinding.create(
                "unknown_required_surface" if required else "unknown_optional_surface",
                "surface kind is not known to the accepted Student contract",
                surface_id=surface.surface_id,
                surface_kind=surface.surface_kind,
            )
            (blockers if required else warnings).append(finding)
            scopes.append(
                TargetScopeCompatibility(
                    surface_id=surface.surface_id,
                    scope_kind=_scope_kind(surface),
                    required=required,
                    status="unevaluated",
                )
            )
            continue
        surface_blocked = False
        if surface.surface_kind not in supported_kinds:
            surface_blocked = True
            _surface_finding(
                code="unsupported_surface_kind",
                message="surface kind is not supported by the profile",
                surface=surface,
                required=required,
                blockers=blockers,
                warnings=warnings,
            )
        if (surface.surface_kind, surface.schema_version) not in supported_schemas:
            surface_blocked = True
            _surface_finding(
                code="unsupported_surface_schema",
                message="surface schema is not supported by the profile",
                surface=surface,
                required=required,
                blockers=blockers,
                warnings=warnings,
            )
        scope_kind = _scope_kind(surface)
        scope_supported = scope_kind in supported_scopes
        scopes.append(
            TargetScopeCompatibility(
                surface_id=surface.surface_id,
                scope_kind=scope_kind,
                required=required,
                status=(
                    "pass"
                    if scope_supported
                    else ("fail" if required else "unevaluated")
                ),
            )
        )
        if not scope_supported:
            surface_blocked = True
            finding = CompatibilityFinding.create(
                "unsupported_target_scope",
                "surface target scope is not supported by the profile",
                surface_id=surface.surface_id,
                scope_kind=scope_kind,
            )
            (blockers if required else warnings).append(finding)
        (unsupported if surface_blocked else supported).append(surface.surface_id)
    return tuple(supported), tuple(unsupported), tuple(scopes)


def _surface_finding(
    *,
    code: str,
    message: str,
    surface: AvailableSurface,
    required: bool,
    blockers: list[CompatibilityFinding],
    warnings: list[CompatibilityFinding],
) -> None:
    finding = CompatibilityFinding.create(
        code,
        message,
        surface_id=surface.surface_id,
        surface_kind=surface.surface_kind,
        schema_version=surface.schema_version,
    )
    (blockers if required else warnings).append(finding)


def _check_dimension(
    *,
    dimension: str,
    artifact_value: int | None,
    profile_limit: int | None,
    exceeds_code: str,
    unevaluated_code: str,
    blockers: list[CompatibilityFinding],
) -> DimensionCompatibility:
    if artifact_value is None or profile_limit is None:
        blockers.append(
            CompatibilityFinding.create(
                unevaluated_code,
                f"{dimension} compatibility is not declared by the profile",
                artifact_value=artifact_value,
                profile_limit=profile_limit,
            )
        )
        return DimensionCompatibility("unevaluated", artifact_value, profile_limit)
    if artifact_value > profile_limit:
        blockers.append(
            CompatibilityFinding.create(
                exceeds_code,
                f"artifact {dimension} exceeds the profile limit",
                artifact_value=artifact_value,
                profile_limit=profile_limit,
            )
        )
        return DimensionCompatibility("fail", artifact_value, profile_limit)
    return DimensionCompatibility("pass", artifact_value, profile_limit)


def _check_tokenizer(
    defaults: StudentRunDefaults,
    profile: StudentCapabilityProfile,
    blockers: list[CompatibilityFinding],
    warnings: list[CompatibilityFinding],
) -> None:
    tokenizer_id = defaults.artifact_facts.tokenizer_id
    if profile.supported_tokenizer_ids is None:
        warnings.append(
            CompatibilityFinding.create(
                "tokenizer_architecture_binding_deferred",
                "tokenizer identity is present but architecture binding is deferred",
                tokenizer_id=tokenizer_id,
            )
        )
    elif tokenizer_id not in profile.supported_tokenizer_ids:
        blockers.append(
            CompatibilityFinding.create(
                "unsupported_tokenizer",
                "artifact tokenizer is not supported by the profile",
                tokenizer_id=tokenizer_id,
            )
        )


def _check_plan(
    defaults: StudentRunDefaults,
    profile: StudentCapabilityProfile,
    blockers: list[CompatibilityFinding],
) -> PlanCompatibility:
    plan = defaults.recommended_training_plan
    if not plan:
        return PlanCompatibility(
            "not_present",
            0,
            profile.supports_ordered_passes,
            profile.supports_checkpoint_boundaries,
            profile.supports_pass_prerequisites,
        )
    failed = False
    surface_ids = {surface.surface_id for surface in defaults.available_surfaces}
    if not profile.supports_ordered_passes:
        failed = True
        blockers.append(
            CompatibilityFinding.create(
                "unsupported_training_plan",
                "profile does not support ordered surface passes",
                feature="ordered_passes",
            )
        )
    if any(item.checkpoint_after for item in plan) and not (
        profile.supports_checkpoint_boundaries
    ):
        failed = True
        blockers.append(
            CompatibilityFinding.create(
                "unsupported_training_plan",
                "profile does not support checkpoint boundary metadata",
                feature="checkpoint_boundaries",
            )
        )
    if any(item.prerequisites for item in plan) and not (
        profile.supports_pass_prerequisites
    ):
        failed = True
        blockers.append(
            CompatibilityFinding.create(
                "unsupported_training_plan",
                "profile does not support pass prerequisites",
                feature="pass_prerequisites",
            )
        )
    completed_surfaces: set[str] = set()
    for item in plan:
        if item.surface_id not in surface_ids:
            failed = True
            blockers.append(
                CompatibilityFinding.create(
                    "unsupported_training_plan",
                    "training plan references an unavailable surface",
                    pass_id=item.pass_id,
                    surface_id=item.surface_id,
                )
            )
        missing_prerequisites = set(item.prerequisites) - completed_surfaces
        if missing_prerequisites:
            failed = True
            blockers.append(
                CompatibilityFinding.create(
                    "unsupported_training_plan",
                    "training pass prerequisites are not satisfied in order",
                    pass_id=item.pass_id,
                    missing_prerequisites=tuple(sorted(missing_prerequisites)),
                )
            )
        completed_surfaces.add(item.surface_id)
    return PlanCompatibility(
        "fail" if failed else "pass",
        len(plan),
        profile.supports_ordered_passes,
        profile.supports_checkpoint_boundaries,
        profile.supports_pass_prerequisites,
    )


def _scope_kind(surface: AvailableSurface) -> str:
    value = surface.target_scope.get("kind", "unspecified")
    return "unspecified" if value is None else str(value)


def _artifact_identity(defaults: StudentRunDefaults) -> CompatibilityArtifactIdentity:
    facts = defaults.artifact_facts
    return CompatibilityArtifactIdentity(
        contract_family=facts.contract_family,
        artifact_kind=facts.artifact_kind,
        cover_page_version=facts.cover_page_version,
        tome_version=facts.tome_version,
        source_artifact_type=facts.source_artifact_type,
        producer_validation_status=facts.producer_validation_status,
        contract_validation_status=facts.contract_validation_status,
    )


def _deduplicate_findings(
    findings: list[CompatibilityFinding],
) -> tuple[CompatibilityFinding, ...]:
    result: list[CompatibilityFinding] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        key = (finding.code, finding.message, repr(finding.details))
        if key not in seen:
            result.append(finding)
            seen.add(key)
    return tuple(result)


def _resolved_capability_warning(warning: str, supported: set[str]) -> bool:
    prefix = "unsupported_required_capability: "
    return warning.startswith(prefix) and warning.removeprefix(prefix) in supported
