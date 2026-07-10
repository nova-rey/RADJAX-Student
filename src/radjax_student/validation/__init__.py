"""Student-side validation boundary.

This package is reserved for compatibility and readiness checks that decide
whether this runtime can consume a valid Contract artifact.
"""

from radjax_student.validation.compatibility import (
    evaluate_student_compatibility,
    evaluate_tome_path_compatibility,
    metadata_inspection_only_profile,
)
from radjax_student.validation.compatibility_models import (
    CompatibilityArtifactIdentity,
    CompatibilityFinding,
    DimensionCompatibility,
    PlanCompatibility,
    StudentCapabilityProfile,
    StudentCompatibilityReport,
    TargetScopeCompatibility,
)
from radjax_student.validation.default_models import (
    ArtifactRunFacts,
    AvailableSurface,
    CorridorSurfaceFacts,
    ExemplarSurfaceFacts,
    RecommendedPass,
    StudentRunDefaults,
)
from radjax_student.validation.run_defaults import (
    infer_run_defaults,
    infer_run_defaults_from_tome,
)

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
    "infer_run_defaults",
    "infer_run_defaults_from_tome",
    "metadata_inspection_only_profile",
]
