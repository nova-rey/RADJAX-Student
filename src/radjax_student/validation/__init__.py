"""Student-side validation boundary.

This package is reserved for compatibility and readiness checks that decide
whether this runtime can consume a valid Contract artifact.
"""

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
    "CorridorSurfaceFacts",
    "ExemplarSurfaceFacts",
    "RecommendedPass",
    "StudentRunDefaults",
    "infer_run_defaults",
    "infer_run_defaults_from_tome",
]
