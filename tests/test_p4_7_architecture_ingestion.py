"""Bounded P4.7 checks for architecture-plugin ingestion isolation."""

from __future__ import annotations

import shutil
from pathlib import Path

from radjax_student.validation.architecture_audit import (
    P4_INGESTION_SCHEMA,
    build_phase4_architecture_ingestion_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def _copied_source(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "src", repository / "src")
    return repository


def test_current_production_source_has_exactly_the_p4_ingestion_surface() -> None:
    audit = build_phase4_architecture_ingestion_audit(ROOT)

    assert audit["status"] == "pass"
    assert audit["blockers"] == []
    assert audit["schema_version"] == P4_INGESTION_SCHEMA
    assert [item["change"] for item in audit["approved_generic_changes"]] == [
        "sparse categorical cross-entropy objective",
        "runtime-owned initialization-key materializer",
        "runtime-supplied initialization material on ArchitectureInitRequest",
    ]


def test_p4_audit_rejects_literal_contamination_and_second_registration(
    tmp_path: Path,
) -> None:
    repository = _copied_source(tmp_path)
    source = repository / "src" / "radjax_student"
    (source / "learning" / "p4_probe.py").write_text(
        "from radjax_student.architecture.rwkv7_reference import RWKV7ReferencePlugin\n"
        "if architecture_id == 'radjax.architecture.rwkv7_reference':\n"
        "    pass\n",
        encoding="utf-8",
    )
    (source / "architecture" / "rwkv7_reference" / "p4_probe.py").write_text(
        "from radjax_student.validation import architecture_audit\n",
        encoding="utf-8",
    )
    registration = source / "architecture" / "second_reference" / "plugin.py"
    registration.parent.mkdir()
    registration.write_text(
        "def register(registry, plugin):\n    registry.register(plugin)\n",
        encoding="utf-8",
    )

    audit = build_phase4_architecture_ingestion_audit(repository)
    codes = {blocker["code"] for blocker in audit["blockers"]}

    assert {
        "generic_owner_imports_rwkv_plugin",
        "generic_owner_mentions_rwkv",
        "rwkv_plugin_imports_validation",
        "unexpected_architecture_registration",
    } <= codes
    assert (
        "src/radjax_student/architecture/second_reference/plugin.py"
        in audit["reviewed_source_paths"]
    )


def test_p4_audit_requires_the_declared_rwkv_plugin(tmp_path: Path) -> None:
    repository = _copied_source(tmp_path)
    shutil.rmtree(
        repository / "src" / "radjax_student" / "architecture" / "rwkv7_reference"
    )

    audit = build_phase4_architecture_ingestion_audit(repository)

    assert audit["status"] == "blocked"
    assert [blocker["code"] for blocker in audit["blockers"]] == ["rwkv_plugin_missing"]


def test_p4_audit_rejects_a_second_registration_in_the_allowed_module(
    tmp_path: Path,
) -> None:
    repository = _copied_source(tmp_path)
    registration = (
        repository
        / "src"
        / "radjax_student"
        / "architecture"
        / "rwkv7_reference"
        / "registration.py"
    )
    registration.write_text(
        registration.read_text(encoding="utf-8")
        + "\n\ndef register_second(registry, plugin):\n    registry.register(plugin)\n",
        encoding="utf-8",
    )

    codes = {
        blocker["code"]
        for blocker in build_phase4_architecture_ingestion_audit(repository)["blockers"]
    }

    assert "unexpected_architecture_registration" in codes
