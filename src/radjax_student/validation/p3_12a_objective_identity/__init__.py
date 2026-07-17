"""P3.12A validation contracts; passive import is JAX-free."""

from radjax_student.validation.p3_12a_objective_identity.models import (
    SCHEMA_VERSION,
    ObjectiveIdentityProof,
    ObjectiveProofCase,
    build_receipt,
    digest,
    validate_receipt,
)

__all__ = [
    "ObjectiveIdentityProof",
    "ObjectiveProofCase",
    "SCHEMA_VERSION",
    "build_receipt",
    "digest",
    "validate_receipt",
]
