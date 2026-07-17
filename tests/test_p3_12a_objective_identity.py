"""P3.12A objective identity contracts and public-conveyor evidence."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from radjax_student.contracts import (
    ObjectiveCapabilityProfile,
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveIdentity,
    ResolvedObjectiveSelection,
)
from radjax_student.learning import ObjectiveScope
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    ObjectiveRegistry,
    build_default_objective_registry,
)
from radjax_student.objectives.jax import MeanSquaredErrorObjective
from radjax_student.objectives.legacy import (
    HISTORICAL_MSE_ALIASES,
    resolve_historical_objective_alias,
)
from radjax_student.validation.architecture_audit import build_architecture_audit
from radjax_student.validation.p3_12a_objective_identity.documentation import (
    check_documentation,
)
from radjax_student.validation.p3_12a_objective_identity.models import (
    ObjectiveIdentityProof,
    ObjectiveProofCase,
    build_receipt,
    digest,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


class _AlternativeMseObjective(MeanSquaredErrorObjective):
    pass


class _ProfileMismatchObjective(MeanSquaredErrorObjective):
    def capability_profile(self):
        return ObjectiveCapabilityProfile(
            ObjectiveIdentity("radjax.objective.foreign", "1"),
            ("objective.jax_execution_v1",),
            ("prediction",),
            ("targets.y",),
            "radjax.objective.mean_squared_error.metrics.v1",
            ("objective.mse",),
        )


def _selection():
    registry = build_default_objective_registry()
    selection = registry.select(CANONICAL_MSE_IDENTITY)
    config = ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"})
    resolved = ResolvedObjectiveSelection(ObjectiveScope(), "final_output")
    descriptor = registry.execution_descriptor(
        selection=selection, config=config, resolved_selection=resolved
    )
    return registry, selection, config, resolved, descriptor


def test_neutral_objective_contract_and_registry_imports_do_not_load_jax():
    code = (
        "import sys; import radjax_student.contracts.objective; "
        "import radjax_student.objectives; "
        "assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registry_canonical_identity_descriptor_and_aliases_are_one_authority():
    registry, selection, config, resolved, descriptor = _selection()
    assert selection.is_registry_selected
    assert descriptor.identity == CANONICAL_MSE_IDENTITY
    assert descriptor.config_digest == config.digest
    assert descriptor.resolved_surface_identity == resolved.digest
    for alias in HISTORICAL_MSE_ALIASES:
        translated = resolve_historical_objective_alias(
            source_alias=alias, registry=registry, resolved_selection=resolved
        )
        assert translated.selection is selection
        assert translated.config == config
        assert translated.descriptor == descriptor
        assert translated.descriptor.identity.objective_id != alias or alias == (
            CANONICAL_MSE_IDENTITY.objective_id
        )


def test_registry_rejects_profile_and_implementation_identity_drift():
    registry = ObjectiveRegistry()
    with pytest.raises(ObjectiveContractError, match="objective_identity_mismatch"):
        registry.register(_ProfileMismatchObjective())

    registry.register(_AlternativeMseObjective())
    with pytest.raises(
        ObjectiveContractError, match="objective_implementation_identity_mismatch"
    ):
        registry.register(MeanSquaredErrorObjective())


def test_objective_contracts_reject_duplicate_metrics_and_config_descriptor_drift():
    with pytest.raises(ObjectiveContractError, match="objective_metric_invalid"):
        ObjectiveCapabilityProfile(
            CANONICAL_MSE_IDENTITY,
            ("objective.jax_execution_v1",),
            ("prediction",),
            ("targets.y",),
            "radjax.objective.mean_squared_error.metrics.v1",
            ("objective.mse", "objective.mse"),
        )
    _, _, config, _, descriptor = _selection()
    drifted = replace(descriptor, config_digest="0" * 64)
    assert drifted.config_digest != config.digest


def test_objective_receipt_is_executed_proof_only_and_strictly_parsed():
    _, _, _, _, descriptor = _selection()
    positive = ObjectiveProofCase(
        "positive", "positive", "pass", None, None, "registry", digest({"ok": 1})
    )
    rejected = ObjectiveProofCase(
        "rejected",
        "adversarial",
        "reject",
        "objective_identity_mismatch",
        "objective_identity_mismatch",
        "registry",
        digest({"code": "objective_identity_mismatch"}),
    )
    proof = ObjectiveIdentityProof(
        descriptor,
        (positive,),
        (rejected,),
        digest({"checkpoint": 1}),
        digest({"replay": 1}),
        digest({"report": 1}),
        digest({"audit": 1}),
        ("no_production_architecture",),
    )
    receipt = build_receipt(proof)
    assert validate_receipt(receipt)["status"] == "pass"
    malformed = dict(receipt)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="fields are missing or unknown"):
        validate_receipt(malformed)


def test_dependency_audit_requires_one_registered_objective_authority():
    audit = build_architecture_audit(ROOT)
    assert audit["status"] == "pass", audit["blockers"]


def test_p312a_documentation_status_and_recorded_receipt_are_consistent():
    check = check_documentation(ROOT)
    assert check.ok, check.errors


@pytest.mark.jax
def test_p312a_real_conveyor_receipt_has_all_literal_adversaries(tmp_path):
    from radjax_student.validation.p3_12a_objective_identity.runner_jax import (
        execute_objective_identity_proof,
    )

    proof = execute_objective_identity_proof(tmp_path)
    receipt = build_receipt(proof)
    assert len(proof.positive_cases) == 6
    assert len(proof.adversarial_cases) == 37
    assert receipt["unexpected_pass_count"] == 0
    assert receipt["unexpected_failure_count"] == 0
    assert {item.observed_code for item in proof.adversarial_cases} >= {
        "objective_plugin_invalid",
        "checkpoint_objective_identity_mismatch",
        "replay_objective_identity_mismatch",
    }


@pytest.mark.jax
def test_p312a_objective_receipt_generation_is_byte_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [
        sys.executable,
        "-m",
        "radjax_student.validation.p3_12a_objective_identity",
        "--write",
    ]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for destination in (first, second):
        result = subprocess.run(
            [*command, str(destination)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.jax
def test_p312a_recorded_gate_is_repository_read_only():
    maintained = (
        ROOT / "docs/P3_12A_OBJECTIVE_IDENTITY_RECEIPT.json",
        ROOT / "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md",
        ROOT / "docs/P3_12_FOUNDATION_IDENTITY_POLISH.md",
        ROOT / "docs/P3_11_9_REPLAY_EVIDENCE.json",
        ROOT / "docs/P3_11_10_FINAL_ADVERSARIAL_GATE_RECEIPT.json",
        ROOT / "docs/P3_5_DEPENDENCY_AUDIT.json",
    )
    before = {path: sha256(path.read_bytes()).hexdigest() for path in maintained}
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "radjax_student.validation.p3_12a_objective_identity",
            "--check-recorded",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    after = {path: sha256(path.read_bytes()).hexdigest() for path in maintained}
    assert after == before
