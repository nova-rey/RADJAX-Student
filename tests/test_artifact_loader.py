from pathlib import Path

from radjax_student.artifacts import inspect_teacher_tome
from tests.support.tome_fixtures import write_dense_tome


def test_contract_teacher_tome_artifact_can_be_inspected(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "toy_tome"
    write_dense_tome(artifact_dir)

    manifest = inspect_teacher_tome(artifact_dir)

    assert manifest["producer"] == "radjax-tome"
    assert manifest["artifact_kind"] == "radjax_tome"
