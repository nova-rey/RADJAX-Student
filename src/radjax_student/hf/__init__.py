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
]
