"""Student-side validation boundary.

This package is reserved for compatibility and readiness checks that decide
whether this runtime can consume a valid Contract artifact.
"""

from radjax_student.artifacts.compatibility import (
    evaluate_student_compatibility,
    evaluate_tome_path_compatibility,
    metadata_inspection_only_profile,
)
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
    ArtifactRunFacts,
    AvailableSurface,
    CorridorSurfaceFacts,
    ExemplarSurfaceFacts,
    RecommendedPass,
    StudentRunDefaults,
)
from radjax_student.artifacts.profile_registry import (
    available_profile_ids,
    declaration_test_only_profile,
    resolve_profile,
)
from radjax_student.artifacts.run_defaults import (
    infer_run_defaults,
    infer_run_defaults_from_tome,
)
from radjax_student.validation.architecture_audit import build_architecture_audit

__all__ = [
    "ArtifactRunFacts",
    "AvailableSurface",
    "CompatibilityArtifactIdentity",
    "CompatibilityFinding",
    "CorridorSurfaceFacts",
    "ExemplarSurfaceFacts",
    "DimensionCompatibility",
    "PlanCompatibility",
    "RecommendedPass",
    "StudentRunDefaults",
    "StudentCapabilityProfile",
    "StudentCompatibilityReport",
    "TargetScopeCompatibility",
    "evaluate_student_compatibility",
    "evaluate_tome_path_compatibility",
    "available_profile_ids",
    "build_architecture_audit",
    "declaration_test_only_profile",
    "infer_run_defaults",
    "infer_run_defaults_from_tome",
    "metadata_inspection_only_profile",
    "resolve_profile",
]
