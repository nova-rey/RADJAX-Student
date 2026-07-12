"""Explicit current-status consistency checks for maintained P3.11 documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from radjax_student.validation.p3_11_9_replay.canonical import canonical_digest


@dataclass(frozen=True)
class DocumentationCheck:
    ok: bool
    errors: tuple[str, ...] = ()


_CURRENT_MARKERS = (
    "P3.11.1-P3.11.9 accepted",
    "P3.11.10 next and unstarted",
    "Phase 4 blocked",
)
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
    "docs/P3_5_ARCHITECTURE_INTEGRITY_ROADMAP.md",
    "docs/P3_5_10_FINAL_ARCHITECTURE_INTEGRITY_GATE.md",
)


def check_documentation(
    repository_root: Path, artifact_bytes: bytes
) -> DocumentationCheck:
    """Check only an explicit maintained-current-status document allowlist."""

    errors: list[str] = []
    artifact_digest = canonical_digest({"artifact": artifact_bytes.hex()})
    for relative in _ALLOWLIST:
        path = repository_root / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        content = path.read_text(encoding="utf-8")
        if not all(marker in content for marker in _CURRENT_MARKERS):
            errors.append(f"status_markers:{relative}")
        if "Phase 4 unblocked" in content or "P3.11.10 complete" in content:
            errors.append(f"stale_status:{relative}")
        if "remote CI passed" in content:
            errors.append(f"unsupported_remote_ci_claim:{relative}")
    replay_doc = repository_root / "docs/P3_11_9_DETERMINISTIC_REPLAY.md"
    if replay_doc.is_file() and artifact_digest not in replay_doc.read_text(
        encoding="utf-8"
    ):
        errors.append("artifact_digest:docs/P3_11_9_DETERMINISTIC_REPLAY.md")
    return DocumentationCheck(not errors, tuple(sorted(errors)))


def maintained_paths() -> tuple[str, ...]:
    return _ALLOWLIST


__all__ = ["DocumentationCheck", "check_documentation", "maintained_paths"]
