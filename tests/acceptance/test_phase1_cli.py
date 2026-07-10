from __future__ import annotations

import json

from radjax_student.cli import inspect as inspect_command
from radjax_student.reports import build_doctor_report

from .support import ACCEPTED_FIXTURE_DIGEST, canonical_fixture, run_cli


def test_inspect_human_contains_every_phase1_report_section() -> None:
    code, output, error = run_cli(
        "inspect",
        "--tome",
        str(canonical_fixture()),
    )

    assert code == 1
    assert error == ""
    for expected in (
        "Artifact",
        "Provenance",
        "Validation",
        "Surfaces",
        "Required Capabilities",
        "Recommended Plan",
        "Compatibility",
        "status: FAIL",
        "Blockers",
        "Warnings",
        "Artifact Claims Not Made",
        "Student Claims Not Made",
    ):
        assert expected in output


def test_inspect_json_is_complete_deterministic_and_unstyled() -> None:
    arguments = (
        "inspect",
        "--tome",
        str(canonical_fixture()),
        "--format",
        "json",
    )
    first_code, first_output, first_error = run_cli(*arguments)
    second_code, second_output, second_error = run_cli(*arguments)
    payload = json.loads(first_output)

    assert first_code == second_code == 1
    assert first_error == second_error == ""
    assert first_output == second_output
    assert "\x1b[" not in first_output
    assert set(payload) == {
        "artifact_path",
        "artifact_summary",
        "compatibility",
        "contents",
        "provenance",
        "run_defaults",
        "selected_profile",
        "status",
        "validation",
    }
    assert payload["compatibility"]["artifact_claims_not_made"]
    assert payload["compatibility"]["student_claims_not_made"]


def test_inspect_exit_codes_are_locked(monkeypatch) -> None:
    fixture = str(canonical_fixture())
    pass_code, _, _ = run_cli(
        "inspect",
        "--tome",
        fixture,
        "--profile",
        "declaration_test_only",
    )
    fail_code, _, _ = run_cli("inspect", "--tome", fixture)
    usage_code, _, usage_error = run_cli(
        "inspect",
        "--tome",
        fixture,
        "--profile",
        "missing",
    )

    def explode(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("acceptance sentinel")

    monkeypatch.setattr(inspect_command, "build_inspection_report", explode)
    internal_code, _, internal_error = run_cli("inspect", "--tome", fixture)

    assert pass_code == 0
    assert fail_code == 1
    assert usage_code == 2
    assert "unknown compatibility profile" in usage_error
    assert internal_code == 3
    assert internal_error == "Internal error: RuntimeError: acceptance sentinel\n"
    assert "Traceback" not in internal_error


def test_doctor_proves_pipeline_and_honest_expected_failure() -> None:
    report = build_doctor_report()
    code, output, error = run_cli("doctor", "--format", "json")
    payload = json.loads(output)

    assert code == 0
    assert error == ""
    assert report.status == payload["status"] == "pass"
    assert report.student_package.package == "radjax-student"
    assert report.contract_package.package == "radjax-contract"
    assert report.student_package.version in {None, "0.1.0"}
    assert report.contract_package.version == "0.1.0"
    assert payload["canonical_fixture_helper_available"] is True
    assert payload["actual_fixture_digest"] == ACCEPTED_FIXTURE_DIGEST
    assert payload["fixture_opens"] is True
    assert payload["defaults_inference_succeeds"] is True
    assert payload["compatibility_report_succeeds"] is True
    assert payload["expected_metadata_failure_recognized"] is True
    assert payload["report_serialization_succeeds"] is True
    assert payload["capability_state"] == {
        "metadata_inspection": "available",
        "run_default_inference": "available",
        "compatibility_reporting": "available",
        "payload_loading": "unavailable",
        "training": "unavailable",
        "runtime_execution": "unavailable",
        "hf_export": "unavailable",
    }


def test_output_files_cover_formats_parents_and_overwrite(tmp_path) -> None:
    fixture = str(canonical_fixture())
    human_path = tmp_path / "nested" / "human.txt"
    json_path = tmp_path / "nested" / "json" / "report.json"

    human_code, human_output, human_error = run_cli(
        "inspect",
        "--tome",
        fixture,
        "--output",
        str(human_path),
    )
    json_code, json_output, json_error = run_cli(
        "inspect",
        "--tome",
        fixture,
        "--format",
        "json",
        "--output",
        str(json_path),
    )
    refusal_code, refusal_output, refusal_error = run_cli(
        "inspect",
        "--tome",
        fixture,
        "--output",
        str(human_path),
    )
    overwrite_code, overwrite_output, overwrite_error = run_cli(
        "inspect",
        "--tome",
        fixture,
        "--output",
        str(human_path),
        "--overwrite",
    )

    assert human_code == json_code == overwrite_code == 1
    assert human_error == json_error == overwrite_error == ""
    assert human_output == f"Wrote report: {human_path}\n"
    assert json_output == f"Wrote report: {json_path}\n"
    assert overwrite_output == f"Wrote report: {human_path}\n"
    assert human_path.read_text().startswith("RADJAX-Student Inspect")
    assert json.loads(json_path.read_text())["status"] == "fail"
    assert refusal_code == 2
    assert refusal_output == ""
    assert "output already exists" in refusal_error
