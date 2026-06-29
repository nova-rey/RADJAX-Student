from pathlib import Path

from radjax_contract.artifacts import TeacherTomeManifest
from radjax_contract.io import write_json

from radjax_student.artifacts import inspect_teacher_tome


def test_contract_teacher_tome_artifact_can_be_inspected(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "toy_tome"
    artifact_dir.mkdir()
    write_json(artifact_dir / "manifest.json", TeacherTomeManifest().to_dict())

    manifest = inspect_teacher_tome(artifact_dir)

    assert manifest["producer"] == "radjax-tome"
    assert manifest["schema_name"] == "teacher_tome_v0"
