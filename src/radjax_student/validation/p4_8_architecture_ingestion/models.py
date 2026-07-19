"""Typed JSON contract for the deterministic P4.8 acceptance report."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "radjax.phase4_architecture_ingestion_report.v1"
EQUATION_PARITY_CLAIM = "pinned_numpy_inference_reference"
NON_CLAIM = "not_claimed"
REQUIRED_NON_CLAIMS = {
    "cross_step_bptt",
    "hf_conversion",
    "pretrained_weights",
    "model_quality",
    "performance",
    "multi_device",
    "tpu",
    "teacher_tome_distillation",
    "phase_5",
}
_PLUGIN = {
    "architecture_id": "radjax.architecture.rwkv7_reference",
    "architecture_version": 1,
}
_RUNTIME_CALLABLE = {
    "callable_id": "radjax.learning.generic_jax_step",
    "callable_version": 1,
    "identity_digest": (
        "344911fefb5adfc3e3b840999a566b58e5bcac3cacb0ae140174c2286973d3e7"
    ),
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def derive_status(report: Mapping[str, Any]) -> str:
    """Derive report status from executed, typed evidence only."""

    lifecycle = report["lifecycle"]
    replay = report["checkpoint_replay"]
    passed = (
        report["fixture_provenance"]["verified"],
        lifecycle["eager"]["forward_finite"],
        lifecycle["eager"]["loss_finite"],
        lifecycle["eager"]["gradient_finite"],
        lifecycle["eager"]["finite"],
        lifecycle["eager"]["parameters_changed"],
        lifecycle["eager"]["carry_changed"],
        lifecycle["eager"]["optimizer_advanced"],
        lifecycle["eager"]["key_advanced"],
        lifecycle["jit"]["forward_finite"],
        lifecycle["jit"]["loss_finite"],
        lifecycle["jit"]["gradient_finite"],
        lifecycle["jit"]["finite"],
        lifecycle["jit"]["parameters_changed"],
        lifecycle["jit"]["carry_changed"],
        lifecycle["jit"]["compiled"],
        lifecycle["eager_jit_parameter_equality"],
        lifecycle["prepared_identities"]["distinct"],
        replay["restored_forward_equal"],
        replay["restored_carry_equal"],
        replay["next_step_loss_equal"],
        replay["next_step_parameters_equal"],
        replay["next_step_carry_equal"],
        report["architecture_neutrality"]["status"] == "pass",
    )
    return "pass" if all(passed) else "failed"


def validate_report(report: Mapping[str, Any]) -> None:
    """Reject a report with broadened claims or inconsistent evidence status."""

    required = {
        "schema_version",
        "status",
        "plugin",
        "identities",
        "runtime_callable",
        "lifecycle",
        "checkpoint_replay",
        "architecture_neutrality",
        "fixture_provenance",
        "equation_parity_claim",
        "initialization_parity_claim",
        "training_recipe_parity_claim",
        "weight_file_compatibility",
        "non_claims",
        "evidence_digest",
    }
    if set(report) != required:
        raise ValueError("phase4_report_fields_invalid")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("phase4_report_schema_invalid")
    if report["status"] != derive_status(report):
        raise ValueError("phase4_report_status_invalid")
    if report["plugin"] != _PLUGIN:
        raise ValueError("phase4_report_plugin_identity_invalid")
    if report["runtime_callable"] != _RUNTIME_CALLABLE:
        raise ValueError("phase4_report_runtime_callable_invalid")
    if report["equation_parity_claim"] != EQUATION_PARITY_CLAIM:
        raise ValueError("phase4_report_equation_claim_invalid")
    if report["initialization_parity_claim"] != NON_CLAIM:
        raise ValueError("phase4_report_initialization_claim_invalid")
    if report["training_recipe_parity_claim"] != NON_CLAIM:
        raise ValueError("phase4_report_training_claim_invalid")
    if report["weight_file_compatibility"] is not False:
        raise ValueError("phase4_report_weight_compatibility_invalid")
    non_claims = report["non_claims"]
    if not isinstance(non_claims, Mapping) or set(non_claims) != REQUIRED_NON_CLAIMS:
        raise ValueError("phase4_report_non_claims_invalid")
    if any(value != NON_CLAIM for value in non_claims.values()):
        raise ValueError("phase4_report_non_claims_invalid")
    without_digest = dict(report)
    evidence_digest = without_digest.pop("evidence_digest")
    if evidence_digest != _digest(without_digest):
        raise ValueError("phase4_report_evidence_digest_invalid")


def canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    """Return validated, byte-stable report bytes."""

    validate_report(report)
    return (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
