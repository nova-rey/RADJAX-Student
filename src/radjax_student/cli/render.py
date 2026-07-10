from __future__ import annotations

import json
from pathlib import Path

from radjax_student.reports.doctor import PackageStatus, StudentDoctorReport
from radjax_student.reports.inspection import StudentInspectionReport


class OutputExistsError(ValueError):
    pass


def render_json(report: StudentInspectionReport | StudentDoctorReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def render_inspection_human(
    report: StudentInspectionReport,
    *,
    show_contents: bool = False,
) -> str:
    facts = report.run_defaults.artifact_facts
    compatibility = report.compatibility
    lines = [
        "RADJAX-Student Inspect",
        "",
        "Artifact",
        f"  kind: {facts.artifact_kind}",
        f"  Tome version: {_display(facts.tome_version)}",
        f"  source: {facts.source_artifact_type}",
        f"  teacher: {_display(facts.teacher_model_identity)}",
        f"  tokenizer: {_display(facts.tokenizer_id)}",
        f"  vocab size: {_display(facts.vocab_size)}",
        f"  sequence length: {_display(facts.sequence_length)}",
        f"  examples: {_display(facts.example_count)}",
        "",
        "Provenance",
        f"  producer: {facts.producer_identity}",
        f"  teacher family: {_display(facts.teacher_family)}",
        f"  teacher backend: {_display(facts.teacher_backend)}",
        f"  teacher revision: {_display(facts.teacher_model_revision)}",
        f"  tokenizer hash: {_display(facts.tokenizer_hash)}",
        "",
        "Validation",
        f"  producer: {facts.producer_validation_status.upper()}",
        "  validated by: " + _display(report.validation.get("producer_validated_by")),
        f"  Contract: {facts.contract_validation_status.upper()}",
        "  Student interpretation: "
        + _display(report.validation.get("student_interpretation")),
        "",
        "Surfaces",
    ]
    for surface in report.run_defaults.available_surfaces:
        lines.extend(
            [
                f"  {surface.surface_id}",
                f"    kind: {surface.surface_kind}",
                f"    schema: {surface.schema_version}",
                f"    known: {'yes' if surface.known_surface else 'no'}",
                "    capabilities: "
                + _joined(surface.required_capabilities, empty="none"),
            ]
        )
        if surface.corridor is not None:
            lines.extend(
                [
                    f"    mode policy: {surface.corridor.mode_policy}",
                    f"    modes: {surface.corridor.mode_count}",
                    f"    assignments: {surface.corridor.assignment_count}",
                ]
            )
        if surface.exemplar is not None:
            lines.extend(
                [
                    "    selected examples: "
                    f"{surface.exemplar.selected_exemplar_count}",
                    "    dynamic top-k: "
                    + _mapping_summary(surface.exemplar.dynamic_top_k_metadata),
                ]
            )
    lines.extend(["", "Required Capabilities"])
    lines.extend(f"  - {item}" for item in report.run_defaults.required_capabilities)
    lines.extend(["", "Recommended Plan"])
    if report.run_defaults.recommended_training_plan:
        for item in report.run_defaults.recommended_training_plan:
            lines.append(
                f"  {item.pass_index + 1}. {item.surface_id}"
                f"  checkpoint after: {'yes' if item.checkpoint_after else 'no'}"
            )
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Compatibility",
            f"  profile: {compatibility.profile_id}",
            f"  status: {compatibility.status.upper()}",
        ]
    )
    if report.selected_profile.notes:
        lines.append("  profile notes: " + "; ".join(report.selected_profile.notes))
    lines.extend(["", "Blockers"])
    lines.extend(_finding_lines(compatibility.blockers))
    lines.extend(["", "Warnings"])
    lines.extend(_finding_lines(compatibility.warnings))
    lines.extend(["", "Artifact Claims Not Made"])
    lines.extend(_claim_lines(compatibility.artifact_claims_not_made))
    lines.extend(["", "Student Claims Not Made"])
    lines.extend(_claim_lines(compatibility.student_claims_not_made))
    if show_contents:
        lines.extend(["", "Contents"])
        for ref in report.contents:
            requirement = "required" if ref["required"] else "optional"
            lines.append(
                f"  - {ref['role']}: {ref['path']}"
                f" ({requirement}, {ref['classification']})"
            )
    return "\n".join(lines)


def render_doctor_human(report: StudentDoctorReport) -> str:
    lines = [
        "RADJAX-Student Doctor",
        "",
        f"Status: {report.status.upper()}",
        "",
        "Installation",
        f"  Python: {report.python_version}",
        "  RADJAX-Student: " + _package_summary(report.student_package),
        "  RADJAX-Contract: " + _package_summary(report.contract_package),
        "",
        "Contract Boundary",
        f"  production APIs: {_yes_no(report.contract_apis_importable)}",
        "  canonical fixture helper: "
        + _yes_no(report.canonical_fixture_helper_available),
        f"  canonical fixture: {_yes_no(report.canonical_fixture_available)}",
        f"  fixture digest: {_yes_no(report.fixture_digest_matches)}",
        f"  expected digest: {report.expected_fixture_digest}",
        f"  actual digest: {_display(report.actual_fixture_digest)}",
        f"  fixture opens: {_yes_no(report.fixture_opens)}",
        "",
        "Phase 1 Pipeline",
        f"  defaults inference: {_yes_no(report.defaults_inference_succeeds)}",
        f"  compatibility report: {_yes_no(report.compatibility_report_succeeds)}",
        "  expected metadata-only failure recognized: "
        + _yes_no(report.expected_metadata_failure_recognized),
        f"  report serialization: {_yes_no(report.report_serialization_succeeds)}",
        "",
        "Available Profiles",
    ]
    lines.extend(f"  - {item}" for item in report.available_profiles)
    lines.extend(["", "Current Capability State"])
    lines.extend(
        f"  {name.replace('_', ' ')}: {status}"
        for name, status in report.capability_state.items()
    )
    lines.extend(["", "Blockers"])
    lines.extend(_plain_lines(report.blockers))
    lines.extend(["", "Warnings"])
    lines.extend(_plain_lines(report.warnings))
    lines.extend(["", "Claims Not Made"])
    lines.extend(_claim_lines(report.claims_not_made))
    return "\n".join(lines)


def write_rendered_output(
    path: str | Path,
    content: str,
    *,
    overwrite: bool,
) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise OutputExistsError(
            f"output already exists; pass --overwrite to replace it: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return output_path


def _finding_lines(findings: tuple) -> list[str]:
    if not findings:
        return ["  none"]
    return [f"  - {item.code}: {item.message}" for item in findings]


def _claim_lines(claims: tuple[str, ...]) -> list[str]:
    return _plain_lines(claims)


def _plain_lines(items: tuple[str, ...]) -> list[str]:
    if not items:
        return ["  none"]
    return [f"  - {item}" for item in items]


def _package_summary(package: PackageStatus) -> str:
    version = _display(package.version)
    commit = package.commit
    return version if commit is None else f"{version} ({commit})"


def _mapping_summary(value: object) -> str:
    if not hasattr(value, "items"):
        return _display(value)
    return ", ".join(f"{key}={item}" for key, item in value.items())


def _joined(items: tuple[str, ...], *, empty: str) -> str:
    return ", ".join(items) if items else empty


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _display(value: object) -> str:
    return "unknown" if value is None else str(value)
