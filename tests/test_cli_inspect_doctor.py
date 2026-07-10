from __future__ import annotations

import json
import shutil
import socket
import sys
import urllib.request
from io import StringIO
from pathlib import Path

from radjax_contract.testing import production_tome_fixture_path

from radjax_student.artifacts import targets as target_loading
from radjax_student.cli import inspect as inspect_command
from radjax_student.cli.main import main
from radjax_student.reports import ACCEPTED_FIXTURE_DIGEST
from radjax_student.students.tiny_debug import TinyDebugStudentBackend


def test_inspect_human_reports_pipeline_and_honest_failure() -> None:
    code, output, error = _run_cli(
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
    )

    assert code == 1
    assert error == ""
    assert "kind: radjax_tome" in output
    assert "Provenance" in output
    assert "Validation" in output
    assert "corridor" in output
    assert "exemplar" in output
    assert "Recommended Plan" in output
    assert "status: FAIL" in output
    assert "missing_required_capability" in output
    assert "Artifact Claims Not Made" in output
    assert "Student Claims Not Made" in output


def test_inspect_json_is_complete_and_machine_readable() -> None:
    code, output, error = _run_cli(
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
        "--format",
        "json",
    )

    payload = json.loads(output)
    assert code == 1
    assert error == ""
    assert payload["status"] == "fail"
    assert payload["artifact_summary"]["artifact_kind"] == "radjax_tome"
    assert [
        surface["surface_id"]
        for surface in payload["run_defaults"]["available_surfaces"]
    ] == ["corridor", "exemplar"]
    assert payload["run_defaults"]["recommended_training_plan"]
    assert payload["compatibility"]["blockers"]
    assert payload["compatibility"]["artifact_claims_not_made"]
    assert payload["compatibility"]["student_claims_not_made"]
    assert payload["selected_profile"]["profile_id"] == "metadata_inspection_only"


def test_declaration_test_only_profile_can_pass_without_readiness_claim() -> None:
    code, output, error = _run_cli(
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
        "--profile",
        "declaration_test_only",
        "--format",
        "json",
    )

    payload = json.loads(output)
    assert code == 0
    assert error == ""
    assert payload["status"] == "pass"
    assert payload["selected_profile"]["profile_id"] == "declaration_test_only"
    assert "TEST ONLY" in payload["selected_profile"]["notes"][0]
    assert (
        "capability_declaration_not_implementation_proof"
        in payload["compatibility"]["student_claims_not_made"]
    )


def test_malformed_artifact_returns_stable_artifact_error(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    cover["contents"][0]["sha256"] = "0" * 64
    _write_json(artifact / "cover_page.json", cover)

    code, output, error = _run_cli("inspect", "--tome", str(artifact))

    assert code == 2
    assert output == ""
    assert "Artifact could not be opened." in error
    assert "content_hash_mismatch" in error


def test_unknown_optional_surface_is_a_nonblocking_warning(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    cover = _read_json(artifact / "cover_page.json")
    cover["behavioral_surfaces"].append(_future_surface())
    _write_json(artifact / "cover_page.json", cover)

    code, output, error = _run_cli(
        "inspect",
        "--tome",
        str(artifact),
        "--profile",
        "declaration_test_only",
        "--format",
        "json",
    )

    payload = json.loads(output)
    assert code == 0
    assert error == ""
    assert "future_surface" in payload["compatibility"]["unsupported_surfaces"]
    assert "unknown_optional_surface" in _finding_codes(
        payload["compatibility"]["warnings"]
    )


def test_output_writing_requires_explicit_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "inspection.json"
    arguments = (
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
        "--format",
        "json",
        "--output",
        str(output_path),
    )

    first_code, first_output, first_error = _run_cli(*arguments)
    second_code, second_output, second_error = _run_cli(*arguments)
    overwrite_code, overwrite_output, overwrite_error = _run_cli(
        *arguments,
        "--overwrite",
    )

    assert first_code == 1
    assert first_error == ""
    assert first_output.startswith("Wrote report:")
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "fail"
    assert second_code == 2
    assert second_output == ""
    assert "output already exists" in second_error
    assert overwrite_code == 1
    assert overwrite_error == ""
    assert overwrite_output.startswith("Wrote report:")


def test_show_contents_adds_validated_content_index() -> None:
    code, output, error = _run_cli(
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
        "--show-contents",
    )

    assert code == 1
    assert error == ""
    assert "Contents" in output
    assert "corridor_summary: corridors/corridor_summary.json" in output


def test_doctor_human_verifies_fixture_and_reports_noncapabilities() -> None:
    code, output, error = _run_cli("doctor")

    assert code == 0
    assert error == ""
    assert "Status: PASS" in output
    assert f"expected digest: {ACCEPTED_FIXTURE_DIGEST}" in output
    assert "canonical fixture helper: yes" in output
    assert "fixture digest: yes" in output
    assert "metadata inspection: available" in output
    assert "payload loading: unavailable" in output
    assert "training: unavailable" in output
    assert "expected metadata-only failure recognized: yes" in output


def test_doctor_json_exposes_only_public_profiles() -> None:
    code, output, error = _run_cli("doctor", "--format", "json")

    payload = json.loads(output)
    assert code == 0
    assert error == ""
    assert payload["status"] == "pass"
    assert payload["actual_fixture_digest"] == ACCEPTED_FIXTURE_DIGEST
    assert payload["canonical_fixture_helper_available"] is True
    assert payload["fixture_digest_matches"] is True
    assert payload["available_profiles"] == ["metadata_inspection_only"]
    assert payload["capability_state"]["runtime_execution"] == "unavailable"
    assert "model_quality_not_claimed" in payload["claims_not_made"]


def test_commands_do_not_load_payloads_allocate_models_train_or_use_network(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("execution-only API was called")

    monkeypatch.setattr(target_loading, "load_dense_tome_targets", forbidden)
    monkeypatch.setattr(TinyDebugStudentBackend, "init", forbidden)
    monkeypatch.setattr(
        "radjax_student.training.run_tiny_train_step",
        forbidden,
    )
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    sys.modules.pop("radjax_student.schedules", None)
    sys.modules.pop("radjax_student.runtime", None)

    inspect_code, _, inspect_error = _run_cli(
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
    )
    doctor_code, _, doctor_error = _run_cli("doctor")

    assert inspect_code == 1
    assert inspect_error == ""
    assert doctor_code == 0
    assert doctor_error == ""
    assert "radjax_student.schedules" not in sys.modules
    assert "radjax_student.runtime" not in sys.modules


def test_unknown_profile_is_usage_error() -> None:
    code, output, error = _run_cli(
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
        "--profile",
        "unknown",
    )

    assert code == 2
    assert output == ""
    assert "unknown compatibility profile" in error


def test_unexpected_error_is_rendered_without_traceback(monkeypatch) -> None:
    def explode(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(inspect_command, "build_inspection_report", explode)

    code, output, error = _run_cli(
        "inspect",
        "--tome",
        str(production_tome_fixture_path()),
    )

    assert code == 3
    assert output == ""
    assert error == "Internal error: RuntimeError: controlled failure\n"
    assert "Traceback" not in error


def _run_cli(*arguments: str) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = main(arguments, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "artifact"
    shutil.copytree(production_tome_fixture_path(), destination)
    return destination


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _future_surface() -> dict:
    return {
        "optional_content_roles": [],
        "prerequisites": [],
        "required_capabilities": [],
        "required_content_roles": [],
        "schema_version": "future_surface_v1",
        "semantics": {"future_semantic": "preserved"},
        "surface_id": "future_surface",
        "surface_kind": "future_surface_kind",
        "target_scope": {"kind": "plugin_defined", "plugin": "future"},
    }


def _finding_codes(findings: list[dict]) -> list[str]:
    return [item["code"] for item in findings]
