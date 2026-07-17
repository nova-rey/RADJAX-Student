"""P3.12B HF descriptor authority validation contracts (JAX-free)."""

from radjax_student.validation.p3_12b_hf_descriptor_authority.models import (
    ADVERSARIAL_CASE_COUNT,
    SCHEMA_VERSION,
    HFAdversarialResult,
    HFDescriptorAuthorityProof,
    HFPositiveProof,
    build_receipt,
    validate_receipt,
)

__all__ = [
    "HFDescriptorAuthorityProof",
    "HFAdversarialResult",
    "HFPositiveProof",
    "ADVERSARIAL_CASE_COUNT",
    "SCHEMA_VERSION",
    "build_receipt",
    "validate_receipt",
]
