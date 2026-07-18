"""Explicit P3.12A current-status and receipt consistency validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from radjax_student.validation.p3_12a_objective_identity.models import (
    SCHEMA_VERSION,
    validate_receipt,
)

_STATUS_MARKERS = (
    "P3.11.1-P3.11.10 locally accepted",
    "P3.11 integration closure complete",
    "P3.12A locally accepted",
    "P3.12B locally accepted",
    "P3.12C locally accepted",
    "P3.12D locally accepted",
    "Phase 4 remains unstarted",
    "Phase 4 requires successful required remote base/JAX CI or an explicit "
    "repository-owner waiver",
)
_ALLOWLIST = (
    "README.md",
    "docs/INDEX.md",
    "docs/ROADMAP.md",
    "docs/RADJAX_DEVELOPMENT_ROADMAP.md",
    "docs/RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md",
    "docs/P3_11_INTEGRATION_CLOSURE.md",
    "docs/P3_12_FOUNDATION_IDENTITY_POLISH.md",
    "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md",
)
_NON_CLAIMS = (
    "no production architecture",
    "no Tome payload consumption",
    "no distillation",
    "no Hugging Face export",
    "no accelerator-scale training",
    "no multi-device",
    "no cross-hardware replay",
    "no cross-version replay",
    "no performance",
    "no RadLads-parity",
)
_CONTRACT_DIGEST_LABELS = (
    "Current P3.12A evidence digest:",
    "The current executed evidence digest is",
    "Current P3.12A receipt evidence digest:",
)


@dataclass(frozen=True)
class ObjectiveDocumentationCheck:
    ok: bool
    errors: tuple[str, ...]


def write_contract_evidence_digest(contract: Path, *, evidence_digest: str) -> None:
    """Refresh the contract's generated receipt references from typed evidence."""
    if not contract.is_file():
        raise FileNotFoundError(contract)
    if not re.fullmatch(r"[0-9a-f]{64}", evidence_digest):
        raise ValueError("evidence_digest must be a lowercase SHA-256 digest")
    text = contract.read_text(encoding="utf-8")
    for label in _CONTRACT_DIGEST_LABELS:
        pattern = rf"({re.escape(label)}\s*\n`)[0-9a-f]{{64}}(`)"
        text, replacements = re.subn(pattern, rf"\g<1>{evidence_digest}\g<2>", text)
        if replacements != 1:
            raise ValueError(f"contract digest marker is invalid: {label}")
    contract.write_text(text, encoding="utf-8")


def check_documentation(repository_root: Path) -> ObjectiveDocumentationCheck:
    """Validate an explicit maintained-status set without natural-language scans."""

    errors: list[str] = []
    for relative in _ALLOWLIST:
        path = repository_root / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker not in text for marker in _STATUS_MARKERS):
            errors.append(f"status:{relative}")
        if "remote CI passed" in text or "Phase 4 already begun" in text:
            errors.append(f"unsupported_claim:{relative}")
    contract = repository_root / "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md"
    if contract.is_file():
        text = contract.read_text(encoding="utf-8")
        for non_claim in _NON_CLAIMS:
            if non_claim not in text:
                errors.append(f"non_claim:{non_claim}")
    receipt_path = repository_root / "docs/P3_12A_OBJECTIVE_IDENTITY_RECEIPT.json"
    if receipt_path.is_file():
        try:
            receipt = validate_receipt(
                json.loads(receipt_path.read_text(encoding="utf-8"))
            )
        except (ValueError, json.JSONDecodeError) as error:
            errors.append(f"receipt:{type(error).__name__}")
        else:
            if receipt["schema_version"] != SCHEMA_VERSION:
                errors.append("receipt:schema")
            elif contract.is_file() and receipt[
                "evidence_digest"
            ] not in contract.read_text(encoding="utf-8"):
                errors.append("receipt:digest")
    return ObjectiveDocumentationCheck(not errors, tuple(sorted(errors)))


__all__ = ["ObjectiveDocumentationCheck", "check_documentation"]
