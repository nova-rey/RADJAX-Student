"""Explicit P3.11 closure-document consistency validation."""
# ruff: noqa: E501

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from radjax_student.validation.p3_11_9_replay.canonical import canonical_digest

_ALLOWLIST = (
    "README.md",
    "docs/INDEX.md",
    "docs/ROADMAP.md",
    "docs/RADJAX_DEVELOPMENT_ROADMAP.md",
    "docs/RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md",
    "docs/P3_11_7_CHECKPOINT_V3.md",
    "docs/P3_11_8_STATEFUL_SYSTEMS_PROOF.md",
    "docs/P3_11_9_DETERMINISTIC_REPLAY.md",
    "docs/P3_11_10_FINAL_ADVERSARIAL_GATE.md",
    "docs/P3_11_INTEGRATION_CLOSURE.md",
    "docs/P3_5_ARCHITECTURE_INTEGRITY_ROADMAP.md",
    "docs/P3_5_10_FINAL_ARCHITECTURE_INTEGRITY_GATE.md",
)
_MARKERS = (
    "P3.11.1-P3.11.10 locally accepted",
    "P3.11 integration closure complete",
    "Phase 4 remains unstarted",
    "Phase 4 requires successful required remote base/JAX CI or an explicit repository-owner waiver",
)
_NON_CLAIMS = (
    "no production architecture",
    "no Tome payload consumption",
    "no distillation",
    "no Hugging Face export",
    "no accelerator-scale training",
    "no multi-device proof",
    "no cross-hardware bitwise replay claim",
    "no cross-version bitwise replay claim",
    "no performance claim",
    "no RadLads parity claim",
)
_PROHIBITED_CLAIMS = {
    "P3.11.7 pending": "p311_status_pending_claim",
    "P3.11.8 pending": "p311_status_pending_claim",
    "P3.11.9 pending": "p311_status_pending_claim",
    "P3.11.10 unstarted": "p31110_unstarted_claim",
    "remote CI passed": "unsupported_remote_ci_claim",
    "Phase 4 already begun": "phase4_started_claim",
    "production model trained": "production_model_claim",
    "cross-environment bitwise replay": "cross_environment_replay_claim",
}
_FINAL_GATE_DOCUMENT = "docs/P3_11_10_FINAL_ADVERSARIAL_GATE.md"
_FINAL_GATE_RECEIPT = "docs/P3_11_10_FINAL_ADVERSARIAL_GATE_RECEIPT.json"
_FINAL_GATE_DIGEST_PATTERN = re.compile(
    r"final gate evidence\s+digest\s+`([0-9a-f]{64})`",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentationClosureCheck:
    ok: bool
    errors: tuple[str, ...]

    @property
    def digest(self) -> str:
        return canonical_digest({"ok": self.ok, "errors": list(self.errors)})


class DocumentationClosureError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def check_closure_documentation(
    repository_root: Path,
    *,
    bind_recorded_gate_digest: bool = True,
) -> DocumentationClosureCheck:
    """Validate closure policy and, when requested, its recorded gate digest.

    The gate writer validates the maintained status/non-claim policy while it
    builds a replacement receipt.  The public recorded checker additionally
    binds the human-readable final-gate digest to that completed receipt.
    """

    errors: list[str] = []
    for relative in _ALLOWLIST:
        path = repository_root / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for marker in _MARKERS:
            if marker not in text:
                errors.append(f"status:{relative}:{marker}")
        for claim in _NON_CLAIMS:
            if claim not in normalized:
                errors.append(f"non_claim:{relative}:{claim}")
        for phrase, code in _PROHIBITED_CLAIMS.items():
            if phrase in text:
                errors.append(f"{code}:{relative}")
    if bind_recorded_gate_digest:
        _check_recorded_gate_digest(repository_root, errors)
    return DocumentationClosureCheck(not errors, tuple(sorted(errors)))


def _check_recorded_gate_digest(repository_root: Path, errors: list[str]) -> None:
    document = repository_root / _FINAL_GATE_DOCUMENT
    receipt = repository_root / _FINAL_GATE_RECEIPT
    if not document.is_file() or not receipt.is_file():
        errors.append("final_gate_digest_receipt_missing")
        return
    match = _FINAL_GATE_DIGEST_PATTERN.search(document.read_text(encoding="utf-8"))
    if match is None:
        errors.append("final_gate_digest_documentation_missing")
        return
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        recorded = payload["gate_evidence_digest"]
    except (OSError, TypeError, ValueError, KeyError):
        errors.append("final_gate_digest_receipt_invalid")
        return
    if not isinstance(recorded, str) or match.group(1) != recorded:
        errors.append("final_gate_digest_documentation_stale")


def maintained_paths() -> tuple[str, ...]:
    return _ALLOWLIST


def write_minimal_maintained_documents(repository_root: Path) -> Path:
    """Create a valid, isolated maintained-document tree for an audit.

    Section K experiments exercise the real documentation validator over real
    files.  They deliberately use this stable fixture rather than copies of
    the repository documents: the recorded closure receipt is itself named by
    several maintained documents, and copying that self-referential digest
    would make a documentation mutation alter the receipt it is proving.
    """

    body = "\n".join(
        (
            "# Maintained RADJAX status",
            *_MARKERS,
            *_NON_CLAIMS,
            "",
        )
    )
    for relative in _ALLOWLIST:
        destination = repository_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")
    return repository_root


def require_closure_documentation(check: DocumentationClosureCheck) -> None:
    """Turn a real closure audit result into a stable public rejection."""

    if not check.ok:
        first = check.errors[0] if check.errors else "documentation_validation_rejected"
        code = first.split(":", 1)[0]
        raise DocumentationClosureError(code, "documentation closure validation failed")


__all__ = [
    "DocumentationClosureCheck",
    "DocumentationClosureError",
    "check_closure_documentation",
    "maintained_paths",
    "require_closure_documentation",
    "write_minimal_maintained_documents",
]
