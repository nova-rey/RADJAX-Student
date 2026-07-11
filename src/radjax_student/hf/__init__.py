"""Hugging Face compatibility boundary.

This package is reserved for Hugging Face config, checkpoint, save/load,
inference, and export integration.
"""

from radjax_student.hf.contracts import (
    HFCompatibilityDescriptor,
    HFCompatibilityError,
    HFParameterMapping,
)

__all__ = [
    "HFCompatibilityDescriptor",
    "HFCompatibilityError",
    "HFParameterMapping",
]
