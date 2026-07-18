"""P3.12A objective identity contracts and public-conveyor evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
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
    write_contract_evidence_digest,
)
from radjax_student.validation.p3_12a_objective_identity.models import (
    ObjectiveIdentityProof,
    ObjectiveProofCase,
    build_receipt,
    digest,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
_P312A_DOCUMENTATION_FIXTURE_PATHS = (
    "README.md",
    "docs/INDEX.md",
    "docs/ROADMAP.md",
    "docs/RADJAX_DEVELOPMENT_ROADMAP.md",
    "docs/RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md",
    "docs/P3_11_INTEGRATION_CLOSURE.md",
    "docs/P3_12_FOUNDATION_IDENTITY_POLISH.md",
    "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md",
    "docs/P3_12A_OBJECTIVE_IDENTITY_RECEIPT.json",
)
_P312A_CONTRACT_DIGEST_LABELS = (
    "Current P3.12A evidence digest:",
    "The current executed evidence digest is",
    "Current P3.12A receipt evidence digest:",
)


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


def _copy_p312a_documentation_fixture(destination_root: Path) -> None:
    for relative in _P312A_DOCUMENTATION_FIXTURE_PATHS:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)


def _copy_repository_fixture(destination_root: Path) -> None:
    shutil.copytree(
        ROOT,
        destination_root,
        ignore=shutil.ignore_patterns(".git", ".venv", ".pytest_cache", "__pycache__"),
    )


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


def test_p312a_contract_writer_refreshes_all_generated_digest_references(tmp_path):
    contract = tmp_path / "P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md"
    contract.write_text(
        (ROOT / "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md").read_text(),
        encoding="utf-8",
    )
    receipt = json.loads(
        (ROOT / "docs/P3_12A_OBJECTIVE_IDENTITY_RECEIPT.json").read_text()
    )
    write_contract_evidence_digest(contract, evidence_digest=receipt["evidence_digest"])
    assert contract.read_text(encoding="utf-8").count(receipt["evidence_digest"]) == 3


def test_p312a_documentation_rejects_one_stale_declared_digest(tmp_path: Path) -> None:
    _copy_p312a_documentation_fixture(tmp_path)
    contract = tmp_path / "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md"
    text = contract.read_text(encoding="utf-8")
    text, replacements = re.subn(
        r"(Current P3\.12A receipt evidence digest:\s*\n`)[0-9a-f]{64}(`)",
        rf"\g<1>{'0' * 64}\g<2>",
        text,
    )
    assert replacements == 1
    contract.write_text(text, encoding="utf-8")
    assert check_documentation(tmp_path).errors == ("receipt:digest",)


@pytest.mark.jax
def test_p312a_cli_rejects_each_stale_declared_contract_digest(tmp_path: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for index, label in enumerate(_P312A_CONTRACT_DIGEST_LABELS):
        repository = tmp_path / str(index)
        _copy_p312a_documentation_fixture(repository)
        shutil.copytree(ROOT / "src", repository / "src")
        receipt = repository / "docs/P3_12A_OBJECTIVE_IDENTITY_RECEIPT.json"
        generated = subprocess.run(
            [
                sys.executable,
                "-m",
                "radjax_student.validation.p3_12a_objective_identity",
                "--write",
                str(receipt),
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert generated.returncode == 0, generated.stderr
        contract = repository / "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md"
        text, replacements = re.subn(
            rf"({re.escape(label)}\s*\n`)[0-9a-f]{{64}}(`)",
            rf"\g<1>{'0' * 64}\g<2>",
            contract.read_text(encoding="utf-8"),
        )
        assert replacements == 1
        contract.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "radjax_student.validation.p3_12a_objective_identity",
                "--check-recorded",
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "p312a_documentation_mismatch:receipt:digest" in result.stdout


@pytest.mark.jax
def test_p31110_cli_rejects_stale_p312a_contract_digest(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _copy_repository_fixture(repository)
    contract = repository / "docs/P3_12A_OBJECTIVE_IDENTITY_CONTRACT.md"
    text, replacements = re.subn(
        r"(Current P3\.12A receipt evidence digest:\s*\n`)[0-9a-f]{64}(`)",
        rf"\g<1>{'0' * 64}\g<2>",
        contract.read_text(encoding="utf-8"),
    )
    assert replacements == 1
    contract.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "radjax_student.validation.p3_11_10_gate",
            "--check-recorded",
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "P3.11.10 P3.12A documentation prerequisite failed: receipt:digest" in (
        result.stdout
    )


@pytest.mark.jax
def test_p312a_real_conveyor_receipt_has_all_literal_adversaries(tmp_path):
    from radjax_student.validation.p3_12a_objective_identity.runner_jax import (
        execute_objective_identity_proof,
    )

    proof = execute_objective_identity_proof(tmp_path)
    receipt = build_receipt(proof)
    assert len(proof.positive_cases) == 10
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
