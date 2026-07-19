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


def _audit_source(tmp_path: Path, relative: str, source: str) -> dict:
    path = tmp_path / "src" / "radjax_student" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return build_architecture_audit(tmp_path)


def test_concrete_plugin_local_jax_imports_are_the_only_new_allowance(
    tmp_path: Path,
) -> None:
    audit = _audit_source(
        tmp_path,
        "architecture/example_plugin/plugin.py",
        "def initialize_parameters():\n    import jax\n    import jax.numpy\n",
    )

    assert audit["status"] == "pass"
    assert audit["modules"][0]["concrete_plugin_lazy_jax_imports"] == [
        "jax",
        "jax.numpy",
    ]
    assert audit["modules"][0]["concrete_plugin_rejected_jax_imports"] == []


def test_concrete_plugin_kernel_imports_are_lazy_and_cannot_mask_top_level_jax(
    tmp_path: Path,
) -> None:
    allowed = _audit_source(
        tmp_path,
        "architecture/example_plugin/kernels.py",
        "def rwkv7_step():\n    import jax\n    import jax.numpy\n",
    )
    assert allowed["status"] == "pass"

    blocked = _audit_source(
        tmp_path,
        "architecture/example_plugin/kernels.py",
        "import jax\ndef rwkv7_step():\n    import jax\n",
    )
    assert "forbidden_import" in {item["code"] for item in blocked["blockers"]}


@pytest.mark.parametrize(
    ("relative", "source", "expected_code"),
    (
        (
            "architecture/example_plugin/plugin.py",
            "import jax\n",
            "forbidden_import",
        ),
        (
            "architecture/example_plugin/config.py",
            "def initialize_parameters():\n    import jax\n",
            "forbidden_import",
        ),
        (
            "architecture/models.py",
            "def initialize_parameters():\n    import jax\n",
            "forbidden_import",
        ),
        (
            "architecture/testing.py",
            "def initialize_parameters():\n    import jax\n",
            "forbidden_import",
        ),
        (
            "architecture/example_plugin/__init__.py",
            "def initialize_parameters():\n    import jax\n",
            "forbidden_import",
        ),
        (
            "architecture/example_plugin/plugin.py",
            "def initialize_parameters():\n    import numpy\n",
            "architecture_imports_numpy_model_math",
        ),
        (
            "architecture/example_plugin/plugin.py",
            "def initialize_parameters():\n    import torch\n",
            "forbidden_import",
        ),
    ),
)
def test_concrete_plugin_jax_allowance_does_not_extend_to_other_boundaries(
    tmp_path: Path, relative: str, source: str, expected_code: str
) -> None:
    audit = _audit_source(tmp_path, relative, source)

    assert expected_code in {item["code"] for item in audit["blockers"]}
