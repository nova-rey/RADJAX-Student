"""Hugging Face compatibility boundary.

This package is reserved for Hugging Face config, checkpoint, save/load,
inference, and export integration.
"""

from radjax_student.hf.contracts import (
    HFArchitectureProjection,
    HFCompatibilityDescriptor,
    HFCompatibilityError,
    HFParameterMapping,
    HFParameterProjection,
    HFPreservationReference,
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
)
from radjax_student.hf.language_projection import (
    HF_LANGUAGE_PROJECTION_V1,
    HFLanguageProjectionV1,
    project_hf_language_binding,
)

__all__ = [
    "HFArchitectureProjection",
    "HFCompatibilityDescriptor",
    "HFCompatibilityError",
    "HFParameterMapping",
    "HFParameterProjection",
    "HFPreservationReference",
    "HFSpecialTokenIdentity",
    "HFTokenizerIdentity",
    "HFVocabularyIdentity",
    "HF_LANGUAGE_PROJECTION_V1",
    "HFLanguageProjectionV1",
    "project_hf_language_binding",
]
