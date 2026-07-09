from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_student.artifacts import TomeArtifactView, open_tome_artifact

REQUIRED_FROM_USER: dict[str, None] = {
    "student_architecture": None,
    "student_size_or_config": None,
    "training_budget": None,
    "output_dir": None,
}

UNRESOLVED_BY_PHASE: dict[str, str] = {
    "runtime_backend": "phase_2_or_later",
    "optimizer": "phase_3_or_later",
    "schedule_policy": "phase_3_or_later",
    "hf_export_details": "phase_5_or_later",
    "evaluation_policy": "phase_5_or_later",
}

CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_not_run",
    "student_architecture_not_selected",
    "runtime_not_selected",
    "compatibility_not_passed",
    "model_quality_not_claimed",
    "hf_export_not_ready",
    "radlads_parity_not_measured",
)


@dataclass(frozen=True)
class StudentRunDefaults:
    inferred_from_tome: dict[str, Any]
    required_from_user: dict[str, None]
    unresolved_by_phase: dict[str, str]
    warnings: tuple[str, ...]
    claims_not_made: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inferred_from_tome": dict(self.inferred_from_tome),
            "required_from_user": dict(self.required_from_user),
            "unresolved_by_phase": dict(self.unresolved_by_phase),
            "warnings": list(self.warnings),
            "claims_not_made": list(self.claims_not_made),
        }


def infer_run_defaults(view: TomeArtifactView) -> StudentRunDefaults:
    defaults = view.inferred_defaults
    inferred = {
        "teacher_id": defaults.teacher_id,
        "teacher_family": defaults.teacher_family,
        "teacher_backend": defaults.teacher_backend,
        "tokenizer_id": defaults.tokenizer_id,
        "vocab_size": defaults.vocab_size,
        "sequence_length": view.sequence_length,
        "record_count": view.record_count,
        "payload_format": view.payload_format.value,
        "compression_family": defaults.compression_family,
        "requires_reconstruction": defaults.requires_reconstruction,
        "expected_adapter_family": defaults.adapter_family,
        "artifact_role": defaults.role,
    }
    return StudentRunDefaults(
        inferred_from_tome=inferred,
        required_from_user=dict(REQUIRED_FROM_USER),
        unresolved_by_phase=dict(UNRESOLVED_BY_PHASE),
        warnings=tuple(view.warnings),
        claims_not_made=CLAIMS_NOT_MADE,
    )


def infer_run_defaults_from_tome(path: str | Path) -> StudentRunDefaults:
    return infer_run_defaults(open_tome_artifact(path))
