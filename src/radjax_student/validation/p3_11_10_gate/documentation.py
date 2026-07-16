"""Explicit P3.11 closure-document consistency validation."""
# ruff: noqa: E501

from __future__ import annotations

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
    "Phase 4 next and unstarted",
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


def check_closure_documentation(repository_root: Path) -> DocumentationClosureCheck:
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
        if "remote CI passed" in text or "Phase 4 already begun" in text:
            errors.append(f"unsupported_claim:{relative}")
    return DocumentationClosureCheck(not errors, tuple(sorted(errors)))


def maintained_paths() -> tuple[str, ...]:
    return _ALLOWLIST


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
]
