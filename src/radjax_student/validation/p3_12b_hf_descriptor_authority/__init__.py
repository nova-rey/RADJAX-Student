"""P3.12B HF descriptor authority validation contracts (JAX-free)."""

from radjax_student.validation.p3_12b_hf_descriptor_authority.models import (
    SCHEMA_VERSION,
    HFDescriptorAuthorityProof,
    HFProofCase,
    build_receipt,
    validate_receipt,
)

__all__ = [
    "HFDescriptorAuthorityProof",
    "HFProofCase",
    "SCHEMA_VERSION",
    "build_receipt",
    "validate_receipt",
]
