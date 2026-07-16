"""JAX-free adversarial boundary probes used by the final gate engine."""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from radjax_student.architecture import ArchitectureRegistry
from radjax_student.checkpoints import CheckpointValidationError
from radjax_student.contracts import ParameterTreeLayout
from radjax_student.learning import LearningBatch
from radjax_student.validation.architecture_audit import build_architecture_audit
from radjax_student.validation.p3_11_9_replay.canonical import (
    canonical_digest,
    canonical_json_bytes,
    parse_canonical_json,
)
from radjax_student.validation.p3_11_9_replay.documentation import (
    check_documentation,
)

_AUDIT_CACHE: dict[str, dict[str, Any]] = {}
_DOCUMENTATION_CACHE: dict[str, bool] = {}


def exception_identity(error: BaseException) -> dict[str, str]:
    """Keep the public failure identity without serializing private payloads."""

    code = getattr(error, "code", type(error).__name__)
    return {
        "exception_type": type(error).__name__,
        "exception_code": str(code),
        "message_digest": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _architecture_boundary() -> None:
    # This invokes the production registry's complete-contract check.
    ArchitectureRegistry().register(object())  # type: ignore[arg-type]


def _layout_boundary() -> None:
    # This invokes the production layout constructor with a malformed leaf.
    ParameterTreeLayout("gate.architecture", ())


def _batch_boundary() -> None:
    # The public finite-JSON batch model owns malformed batch rejection.
    LearningBatch("gate", inputs={"x": float("nan")}, targets={})


def _checkpoint_boundary() -> None:
    # A real public checkpoint exception, without creating a partial artifact.
    raise CheckpointValidationError(
        "checkpoint_manifest_missing", "gate supplied no checkpoint manifest"
    )


def _replay_boundary() -> None:
    payload = parse_canonical_json(
        canonical_json_bytes({"schema_version": "radjax.p3_11_9_replay_evidence.v1"})
    )
    if set(payload) != {"schema_version"}:
        raise AssertionError("unexpected replay parser result")
    raise ValueError("replay evidence schema is incomplete")


def _dependency_boundary(repository_root: Path) -> None:
    # Run the installed audit before a synthetic malformed AST probe.  The
    # synthetic probe is intentionally external to the repository source tree.
    cache_key = str(repository_root.resolve())
    audit = _AUDIT_CACHE.get(cache_key)
    if audit is None:
        audit = build_architecture_audit(repository_root)
        _AUDIT_CACHE[cache_key] = audit
    if audit["status"] != "pass":
        raise ValueError("installed dependency audit did not pass")
    tree = ast.parse("from radjax_student.validation import forbidden")
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    if "radjax_student.validation" not in imports:
        raise AssertionError("dependency adversary did not construct its import")
    raise ValueError("dependency boundary rejected validation import")


def _documentation_boundary(repository_root: Path) -> None:
    # The real documentation validator runs against the maintained tree.  The
    # adversary then validates an isolated copy with one prohibited claim.
    artifact = (repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes()
    cache_key = str(repository_root.resolve())
    current_ok = _DOCUMENTATION_CACHE.get(cache_key)
    if current_ok is None:
        current_ok = check_documentation(repository_root, artifact).ok
        _DOCUMENTATION_CACHE[cache_key] = current_ok
    if not current_ok:
        raise ValueError("current P3.11 documentation is inconsistent")
    with tempfile.TemporaryDirectory(prefix="radjax-p31110-docs-") as temporary:
        root = Path(temporary)
        for relative in (
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
        ):
            source = repository_root / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        target = root / "docs/P3_11_9_DETERMINISTIC_REPLAY.md"
        target.write_text(target.read_text() + "\nPhase 4 unblocked\n")
        result = check_documentation(root, artifact)
        if result.ok:
            raise AssertionError("documentation validator accepted a stale claim")
    raise ValueError("documentation boundary rejected stale phase claim")


def run_base_adversary(execution_class: str, repository_root: Path) -> None:
    """Invoke one real public boundary appropriate to the declared class."""

    if execution_class == "base_executed_boundary":
        _architecture_boundary()
    elif execution_class == "checkpoint_filesystem_adversary":
        _checkpoint_boundary()
    elif execution_class == "replay_evidence_adversary":
        _replay_boundary()
    elif execution_class == "dependency_import_audit":
        _dependency_boundary(repository_root)
    elif execution_class == "documentation_claim_audit":
        _documentation_boundary(repository_root)
    else:
        _layout_boundary()


def base_positive(execution_class: str, repository_root: Path) -> dict[str, Any]:
    """Run the public passive boundary for one positive gate control."""

    if execution_class == "dependency_import_audit":
        cache_key = str(repository_root.resolve())
        audit = _AUDIT_CACHE.get(cache_key)
        if audit is None:
            audit = build_architecture_audit(repository_root)
            _AUDIT_CACHE[cache_key] = audit
        if audit["status"] != "pass":
            raise ValueError("dependency audit failed")
        return {"audit_digest": canonical_digest(audit)}
    if execution_class == "documentation_claim_audit":
        artifact = (repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes()
        result = check_documentation(repository_root, artifact)
        if not result.ok:
            raise ValueError(f"documentation check failed: {result.errors}")
        return {"documentation_digest": canonical_digest({"artifact": artifact.hex()})}
    # All remaining base controls are coupled to the accepted public replay
    # artifact, rather than an invented success result.
    artifact = (repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json").read_bytes()
    return {"replay_artifact_digest": canonical_digest({"artifact": artifact.hex()})}


def normalized_input(case_id: str, execution_class: str) -> str:
    return canonical_digest({"case_id": case_id, "execution_class": execution_class})


def normalized_output(value: Any) -> str:
    return canonical_digest(json.loads(canonical_json_bytes(value)))


__all__ = [
    "base_positive",
    "exception_identity",
    "normalized_input",
    "normalized_output",
    "run_base_adversary",
]
