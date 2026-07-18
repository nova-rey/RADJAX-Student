"""P3.12D runtime-callable-identity validation contracts."""

from .inventory import ADVERSARIAL_CASE_IDS, ADVERSARIAL_CASE_SPECS, POSITIVE_CASE_IDS
from .models import (
    AdversarialResult,
    PositiveResult,
    RuntimeCallableIdentityProof,
    RuntimeCallableIdentityReceipt,
    build_receipt,
    validate_receipt,
)

__all__ = [
    "ADVERSARIAL_CASE_IDS",
    "ADVERSARIAL_CASE_SPECS",
    "POSITIVE_CASE_IDS",
    "AdversarialResult",
    "PositiveResult",
    "RuntimeCallableIdentityProof",
    "RuntimeCallableIdentityReceipt",
    "build_receipt",
    "validate_receipt",
]
