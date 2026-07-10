from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_student.artifacts import TomeArtifactView, open_tome_artifact
from radjax_student.validation import (
    StudentCapabilityProfile,
    StudentCompatibilityReport,
    StudentRunDefaults,
    evaluate_student_compatibility,
    infer_run_defaults,
)


@dataclass(frozen=True)
class StudentInspectionReport:
    artifact_path: Path
    artifact_summary: Mapping[str, Any]
    provenance: Mapping[str, Any]
    validation: Mapping[str, Any]
    contents: tuple[Mapping[str, Any], ...]
    run_defaults: StudentRunDefaults
    compatibility: StudentCompatibilityReport
    selected_profile: StudentCapabilityProfile

    @property
    def status(self) -> str:
        return self.compatibility.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact_path": str(self.artifact_path),
            "artifact_summary": _json_value(self.artifact_summary),
            "provenance": _json_value(self.provenance),
            "validation": _json_value(self.validation),
            "contents": [_json_value(item) for item in self.contents],
            "run_defaults": self.run_defaults.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "selected_profile": self.selected_profile.to_dict(),
        }


def build_inspection_report(
    path: str | Path,
    profile: StudentCapabilityProfile,
) -> StudentInspectionReport:
    view = open_tome_artifact(path)
    defaults = infer_run_defaults(view)
    compatibility = evaluate_student_compatibility(view, defaults, profile)
    return StudentInspectionReport(
        artifact_path=view.artifact_dir,
        artifact_summary=defaults.artifact_facts.to_dict(),
        provenance=_provenance(view),
        validation=_validation(view),
        contents=tuple(_content_ref(ref) for ref in view.contents_index),
        run_defaults=defaults,
        compatibility=compatibility,
        selected_profile=profile,
    )


def _provenance(view: TomeArtifactView) -> Mapping[str, Any]:
    return {
        "teacher": view.provenance.teacher,
        "tokenizer": view.provenance.tokenizer,
        "targets": view.provenance.targets,
        "corpus": view.provenance.corpus,
        "teacher_model": view.provenance.teacher_model,
        "producer_lineage": view.provenance.producer_lineage,
    }


def _validation(view: TomeArtifactView) -> Mapping[str, Any]:
    validation = view.validation
    return {
        "producer_status": validation.producer_status,
        "producer_validated_by": validation.producer_validated_by,
        "producer_report_path": validation.producer_report_path,
        "contract_status": validation.contract_status,
        "student_interpretation": validation.student_interpretation,
        "blockers": validation.blockers,
        "warnings": validation.warnings,
    }


def _content_ref(ref: Any) -> Mapping[str, Any]:
    return {
        "role": ref.role,
        "path": ref.path,
        "sha256": ref.sha256,
        "size_bytes": ref.size_bytes,
        "required": ref.required,
        "classification": ref.classification,
        "known_role": ref.known_role,
        "metadata": ref.metadata,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
