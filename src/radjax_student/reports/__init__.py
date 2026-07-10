"""Report and claims boundary.

This package is reserved for run reports, validation reports, evaluation
reports, diagnostics, and explicit claims-not-made sections.
"""

from radjax_student.reports.doctor import (
    ACCEPTED_FIXTURE_DIGEST,
    StudentDoctorReport,
    artifact_tree_digest,
    build_doctor_report,
)
from radjax_student.reports.inspection import (
    StudentInspectionReport,
    build_inspection_report,
)

__all__ = [
    "ACCEPTED_FIXTURE_DIGEST",
    "StudentDoctorReport",
    "StudentInspectionReport",
    "artifact_tree_digest",
    "build_doctor_report",
    "build_inspection_report",
]
