from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_architecture import SCHEMA, build_audit

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "docs" / "P3_5_DEPENDENCY_AUDIT.json"


def test_p3_5_audit_artifact_is_deterministic():
    recorded = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert recorded == build_audit(REPO_ROOT)


def test_p3_5_audit_schema_and_module_inventory_are_complete():
    audit = build_audit(REPO_ROOT)
    source_paths = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src" / "radjax_student").rglob("*.py")
    }
    recorded_paths = {item["path"] for item in audit["modules"]}

    assert audit["schema_version"] == SCHEMA
    assert audit["status"] == "blocked"
    assert audit["module_count"] == len(audit["modules"])
    assert recorded_paths == source_paths
    assert all(item["owner"] for item in audit["modules"])
    assert all(item["classification"] for item in audit["modules"])


def test_p3_5_audit_records_current_architecture_blockers():
    codes = {item["code"] for item in build_audit(REPO_ROOT)["blockers"]}

    assert {
        "transitional_students_namespace",
        "competing_architecture_registries",
        "objective_receives_raw_parameters",
        "forward_result_discarded",
    } <= codes
    assert "root_exports_transitional_students" not in codes
