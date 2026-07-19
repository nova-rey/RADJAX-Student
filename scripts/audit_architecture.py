"""Write the installed P3.5 architecture audit as a repository artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from radjax_student.validation.architecture_audit import (
    SCHEMA,
    build_architecture_audit,
)

build_audit = build_architecture_audit

__all__ = ["SCHEMA", "build_audit", "build_architecture_audit"]


def _source_tree_is_clean(root: Path) -> bool:
    """Evidence must describe a tracked and untracked-clean installed source tree."""
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "src/radjax_student",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout == ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/P3_5_DEPENDENCY_AUDIT.json")
    )
    args = parser.parse_args()
    root = Path.cwd()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not _source_tree_is_clean(root):
        raise RuntimeError(
            "P3.5 dependency audit must be generated from a clean source tree"
        )
    audit = build_architecture_audit(root, accepted_commit=commit)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"P3.5 audit: {audit['status']} ({audit['module_count']} modules)")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
