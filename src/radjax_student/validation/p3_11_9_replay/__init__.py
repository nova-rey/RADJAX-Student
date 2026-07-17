"""Passive contracts for P3.11.9 replay evidence.

This package is deliberately JAX-free at import time.  The optional execution
runner is imported only by the command entry point.
"""

from radjax_student.validation.p3_11_9_replay.models import (
    ArchitectureCarryIdentityEvidence,
    CrossModeComparisonEvidence,
    ExperimentIdentityEvidence,
    HFPreservationEvidence,
    ObjectiveEvidence,
    OptimizerConfigEvidence,
    ReplayArmEvidence,
    ReplayBlocker,
    ReplayModeEvidence,
    ReplayRunEvidence,
    ReplayStepEvidence,
    ReplayVerificationResult,
    RngEvidence,
    RuntimeEvidence,
    StatefulReplayProof,
    StatefulReplayReceipt,
    ToleranceEvidence,
    VerifierEvidence,
)

__all__ = [
    "ReplayArmEvidence",
    "ArchitectureCarryIdentityEvidence",
    "CrossModeComparisonEvidence",
    "ExperimentIdentityEvidence",
    "HFPreservationEvidence",
    "ObjectiveEvidence",
    "OptimizerConfigEvidence",
    "ReplayBlocker",
    "ReplayModeEvidence",
    "ReplayRunEvidence",
    "ReplayStepEvidence",
    "ReplayVerificationResult",
    "RngEvidence",
    "RuntimeEvidence",
    "StatefulReplayProof",
    "StatefulReplayReceipt",
    "ToleranceEvidence",
    "VerifierEvidence",
]
