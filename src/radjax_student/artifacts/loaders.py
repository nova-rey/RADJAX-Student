from __future__ import annotations

from pathlib import Path
from typing import Any

from radjax_contract.io import read_json
from radjax_contract.validation import validate_teacher_tome


def inspect_teacher_tome(path: str | Path) -> dict[str, Any]:
    """Inspect the legacy dense-v0 smoke artifact shape.

    Production artifacts must use ``open_tome_artifact`` and the Contract
    production cover-page API. This helper remains only for existing debug code.
    """

    artifact_dir = Path(path)
    validation = validate_teacher_tome(artifact_dir)
    if not validation.ok:
        raise ValueError("teacher tome does not exist")
    manifest = read_json(artifact_dir / "manifest.json")
    if manifest.get("producer") != "radjax-tome":
        raise ValueError("teacher tome producer must be radjax-tome")
    return manifest
