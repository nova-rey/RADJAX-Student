from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.audit_architecture as audit_writer
from radjax_student.validation.architecture_audit import (
    SCHEMA,
    build_architecture_audit,
    find_dependency_cycles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "docs" / "P3_5_DEPENDENCY_AUDIT.json"


@pytest.mark.parametrize(
    "status",
    (" M src/radjax_student/runtime/inspection.py\0", "?? src/radjax_student/new.py\0"),
)
def test_source_tree_guard_rejects_dirty_or_untracked_source(
    monkeypatch, status: str
) -> None:
    monkeypatch.setattr(
        audit_writer.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=0, stdout=status
        ),
    )

    assert audit_writer._source_tree_is_clean(REPO_ROOT) is False


def test_audit_writer_rejects_unclean_source_before_writing_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit_architecture.py", "--output", str(output)])
    monkeypatch.setattr(audit_writer, "_source_tree_is_clean", lambda root: False)

    def forbidden_build(*args, **kwargs):
        del args, kwargs
        raise AssertionError("audit build must not run for an unclean source tree")

    monkeypatch.setattr(audit_writer, "build_architecture_audit", forbidden_build)

    with pytest.raises(RuntimeError, match="clean source tree"):
        audit_writer.main()
    assert not output.exists()


def test_p3_5_audit_artifact_is_deterministic():
    recorded = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert recorded == build_architecture_audit(
        REPO_ROOT, accepted_commit=recorded["accepted_commit"]
    )
    assert len(recorded["accepted_commit"]) == 40


def test_p3_5_audit_schema_and_module_inventory_are_complete():
    audit = build_architecture_audit(REPO_ROOT)
    source_paths = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src" / "radjax_student").rglob("*.py")
    }
    recorded_paths = {item["path"] for item in audit["modules"]}

    assert audit["schema_version"] == SCHEMA
    assert audit["status"] == "pass"
    assert audit["module_count"] == len(audit["modules"])
    assert recorded_paths == source_paths
    assert all(item["owner"] for item in audit["modules"])
    assert all(item["classification"] for item in audit["modules"])


def test_p3_5_audit_records_current_architecture_blockers():
    codes = {item["code"] for item in build_architecture_audit(REPO_ROOT)["blockers"]}

    assert {
        "objective_receives_raw_parameters",
        "forward_result_discarded",
    }.isdisjoint(codes)
    assert "root_exports_transitional_students" not in codes


def test_p3_5_audit_cycle_reporting_is_deterministic():
    records = (
        {"module": "radjax_student.alpha", "internal_edges": ["radjax_student.beta"]},
        {"module": "radjax_student.beta", "internal_edges": ["radjax_student.alpha"]},
    )
    assert find_dependency_cycles(records) == [
        ["radjax_student.alpha", "radjax_student.beta"]
    ]
