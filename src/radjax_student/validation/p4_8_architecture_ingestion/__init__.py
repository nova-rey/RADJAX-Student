"""P4.8 deterministic architecture-ingestion acceptance evidence."""

from radjax_student.validation.p4_8_architecture_ingestion.models import (
    SCHEMA_VERSION,
    canonical_report_bytes,
)
from radjax_student.validation.p4_8_architecture_ingestion.runner_jax import (
    generate_phase4_report,
    write_phase4_report,
)

__all__ = [
    "SCHEMA_VERSION",
    "canonical_report_bytes",
    "generate_phase4_report",
    "write_phase4_report",
]
