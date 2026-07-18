from __future__ import annotations

from radjax_student.artifacts.compatibility import metadata_inspection_only_profile
from radjax_student.artifacts.compatibility_models import StudentCapabilityProfile

DECLARATION_TEST_ONLY_PROFILE_ID = "declaration_test_only"


def available_profile_ids() -> tuple[str, ...]:
    """Return profiles suitable for normal user-facing selection."""

    return (metadata_inspection_only_profile().profile_id,)


def resolve_profile(profile_id: str) -> StudentCapabilityProfile:
    if profile_id == "metadata_inspection_only":
        return metadata_inspection_only_profile()
    if profile_id == DECLARATION_TEST_ONLY_PROFILE_ID:
        return declaration_test_only_profile()
    raise ValueError(f"unknown compatibility profile: {profile_id}")


def declaration_test_only_profile() -> StudentCapabilityProfile:
    """Declare canonical requirements solely to exercise evaluator/CLI wiring."""

    return StudentCapabilityProfile(
        profile_id=DECLARATION_TEST_ONLY_PROFILE_ID,
        supported_contract_families=("production_v2",),
        supported_tome_versions=(1,),
        supported_cover_page_versions=(2,),
        supported_surface_kinds=("fingerprint_corridor", "selected_exemplar"),
        supported_surface_schemas=(
            ("fingerprint_corridor", "behavioral_surface_v1"),
            ("selected_exemplar", "behavioral_surface_v1"),
        ),
        supported_capabilities=(
            "radjax.corridor.packed_assignments.v1",
            "radjax.corridor.stat_bands.v1",
            "radjax.exemplar.selected_dynamic_topk.v1",
        ),
        supported_target_scopes=("whole_model", "unspecified", "default"),
        max_sequence_length=4,
        max_vocab_size=32,
        supported_tokenizer_ids=("fake-production-tokenizer",),
        notes=("TEST ONLY: declaration proves comparison logic, not implementation",),
    )
