from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.hf import HFCompatibilityError, HFParameterMapping
from radjax_student.learning import LearningIssue
from radjax_student.learning.p3_5_acceptance import (
    FLAGS,
    SCHEMA,
    SECTION_CODES,
    GateCheck,
    P35AcceptanceDependencies,
    P35ArchitectureIntegrityReceipt,
    _check_architecture_objective,
    _check_checkpoint,
    _check_hf,
    _p35_hf_descriptor,
    _run_command,
    _run_jax_contract,
    run_p3_5_architecture_integrity_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]


def _values(**changes):
    result = {name: True for name in FLAGS}
    result.update(changes)
    return result


def _pass(**evidence):
    return GateCheck(True, evidence)


def _audit(_: Path):
    return {"status": "pass", "blockers": [], "module_count": 1}


def _stable_dependencies(**changes):
    dependencies = P35AcceptanceDependencies(
        build_audit=_audit,
        check_architecture_objective=lambda: _pass(architecture=True),
        run_jax_contract=lambda: _pass(jax=True),
        check_namespace_legacy=lambda: (_pass(namespace=True), _pass(legacy=True)),
        check_hf=lambda: _pass(hf=True),
        check_checkpoint=lambda: _pass(checkpoint=True),
        check_docs=lambda _: _pass(docs=True),
        run_command=lambda _, target, section: _pass(target=target, section=section),
        check_import_purity=lambda _: _pass(imports=True),
    )
    return dataclasses.replace(dependencies, **changes)


def test_01_receipt_accepts_complete_passing_evidence():
    receipt = run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies()
    )
    assert receipt.status == "pass"


def test_02_receipt_schema_is_exact():
    with pytest.raises(ValueError, match="schema"):
        P35ArchitectureIntegrityReceipt("wrong", "pass", **_values())


def test_03_receipt_flags_must_be_booleans():
    with pytest.raises(TypeError, match="booleans"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "pass", **_values(import_purity_valid=1)
        )


def test_04_receipt_rejects_pass_with_failed_section():
    with pytest.raises(ValueError, match="status"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "pass", **_values(phase1_regression_valid=False)
        )


def test_05_receipt_rejects_pass_with_blocker():
    blocker = LearningIssue.create("p35_phase1_regression_failed", "failed")
    with pytest.raises(ValueError, match="status"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "pass", blockers=(blocker,), **_values()
        )


def test_06_receipt_rejects_failure_without_blocker():
    with pytest.raises(ValueError, match="requires blocker"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "fail", **_values(phase1_regression_valid=False)
        )


def test_07_receipt_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        P35ArchitectureIntegrityReceipt(SCHEMA, "unknown", **_values())


def test_08_receipt_rejects_non_issue_blocker():
    with pytest.raises(TypeError, match="LearningIssue"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "fail", blockers=("bad",), **_values(phase1_regression_valid=False)
        )


def test_09_receipt_metadata_is_immutable():
    receipt = run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies()
    )
    with pytest.raises(TypeError):
        receipt.metadata["tamper"] = True


def test_10_receipt_json_is_deterministic():
    dependencies = _stable_dependencies()
    first = run_p3_5_architecture_integrity_acceptance(ROOT, dependencies=dependencies)
    second = run_p3_5_architecture_integrity_acceptance(ROOT, dependencies=dependencies)
    assert first.to_json() == second.to_json()


def test_11_dependency_audit_passes_on_current_tree():
    receipt = run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies()
    )
    assert receipt.dependency_boundaries_valid


@pytest.mark.jax
def test_12_architecture_section_executes_surface_and_jaxpr_evidence():
    check = _check_architecture_objective()
    assert check.valid and check.evidence["forward_dots"] == 1


@pytest.mark.jax
def test_13_jax_section_executes_finite_and_carry_evidence():
    check = _run_jax_contract()
    assert check.valid and check.evidence["output_carry_gradient"] == 0.0


def test_14_hf_section_executes_duplicate_and_catalog_evidence():
    check = _check_hf()
    assert check.valid and check.evidence["duplicate_jax_rejected"]
    assert check.evidence["missing_catalog_rejected"]


def test_15_checkpoint_section_executes_round_trip_and_integrity_evidence():
    check = _check_checkpoint()
    assert check.valid and check.evidence["round_trip"]
    assert check.evidence["source_integrity_rejected"]


def test_16_hf_duplicate_jax_path_is_rejected_by_real_constructor():
    from radjax_student.learning.p3_5_acceptance import _hf_fixture

    config, catalog, mappings = _hf_fixture()

    duplicate = (
        mappings[0],
        HFParameterMapping(
            "head.weight",
            ("head", "bias"),
            (1, 1),
            "float32",
            "exportable",
            "head.weight",
            "identity",
        ),
    )
    with pytest.raises(HFCompatibilityError):
        _p35_hf_descriptor(config, catalog, duplicate)


def test_17_hf_duplicate_distribution_key_is_rejected_by_real_constructor():
    from radjax_student.learning.p3_5_acceptance import _hf_fixture

    config, catalog, mappings = _hf_fixture()
    duplicate = (
        mappings[0],
        HFParameterMapping(
            "head.weight",
            ("head", "weight"),
            (1, 1),
            "float32",
            "exportable",
            "head.bias",
            "identity",
        ),
    )
    with pytest.raises(HFCompatibilityError):
        _p35_hf_descriptor(config, catalog, duplicate)


def test_18_hf_runtime_layout_name_is_rejected_by_real_constructor():
    with pytest.raises(HFCompatibilityError):
        HFParameterMapping(
            "head.weight",
            ("mesh", "weight"),
            (1, 1),
            "float32",
            "exportable",
            "head.weight",
            "identity",
        )


def test_19_hf_constructor_preserves_valid_bijection():
    assert _check_hf().evidence["mapping_count"] == 2


def test_20_checkpoint_v2_descriptor_is_scalar_mapping():
    assert _check_checkpoint().evidence["scalar_descriptor"]


def test_21_checkpoint_never_emits_tensor_payload_in_v2():
    assert _check_checkpoint().evidence["tensor_not_emitted"]


def test_22_checkpoint_rejects_continuation_hf_conflation():
    assert _check_checkpoint().evidence["role_rejected"]


def test_23_checkpoint_manifest_excludes_runtime_handles():
    assert _check_checkpoint().evidence["no_runtime_handles"]


def test_24_phase_command_failure_uses_section_specific_blocker_code():
    check = _run_command(ROOT, "not-a-test-target", "phase1_regression_valid")
    assert (
        not check.valid and check.issue.code == SECTION_CODES["phase1_regression_valid"]
    )


def test_25_phase2_command_failure_uses_section_specific_blocker_code():
    check = _run_command(ROOT, "not-a-test-target", "phase2_regression_valid")
    assert (
        not check.valid and check.issue.code == SECTION_CODES["phase2_regression_valid"]
    )


def test_26_phase3_command_uses_real_public_gate():
    check = _run_command(ROOT, "phase3_module", "phase3_regression_valid")
    assert check.valid and check.evidence["returncode"] == 0


def test_27_replay_detects_actual_second_pass_divergence():
    calls = 0

    def changing_command(_, target, section):
        nonlocal calls
        calls += 1
        return _pass(target=target, section=section, invocation=calls)

    receipt = run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies(run_command=changing_command)
    )
    assert not receipt.deterministic_replay_valid
    assert receipt.blockers[-1].code == SECTION_CODES["deterministic_replay_valid"]


def test_28_replay_runs_each_phase_command_twice():
    calls = []

    def observed(_, target, section):
        calls.append((target, section))
        return _pass(target=target, section=section)

    run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies(run_command=observed)
    )
    assert calls == calls[:3] * 2


def test_29_replay_runs_architecture_check_twice():
    calls = []

    def observed():
        calls.append("architecture")
        return _pass(architecture=True)

    run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies(check_architecture_objective=observed)
    )
    assert calls == ["architecture", "architecture"]


def test_30_replay_runs_hf_check_twice():
    calls = []

    def observed():
        calls.append("hf")
        return _pass(hf=True)

    run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies(check_hf=observed)
    )
    assert calls == ["hf", "hf"]


def test_31_replay_runs_checkpoint_check_twice():
    calls = []

    def observed():
        calls.append("checkpoint")
        return _pass(checkpoint=True)

    run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies(check_checkpoint=observed)
    )
    assert calls == ["checkpoint", "checkpoint"]


def test_32_replay_uses_exact_canonical_evidence_not_hashability():
    calls = []

    def changing_hf():
        calls.append(len(calls))
        return _pass(sequence=len(calls))

    receipt = run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies(check_hf=changing_hf)
    )
    assert receipt.status == "fail" and not receipt.deterministic_replay_valid


def test_33_import_purity_is_a_separate_gate_flag():
    receipt = run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=_stable_dependencies()
    )
    assert receipt.import_purity_valid


@pytest.mark.jax
def test_34_real_gate_passes_with_replayed_evidence():
    receipt = run_p3_5_architecture_integrity_acceptance(ROOT)
    assert receipt.status == "pass" and receipt.deterministic_replay_valid


@pytest.mark.jax
def test_35_recorded_receipt_matches_real_gate():
    recorded = json.loads(
        (ROOT / "docs" / "P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json").read_text()
    )
    assert recorded == run_p3_5_architecture_integrity_acceptance(ROOT).to_dict()


@pytest.mark.jax
def test_36_cli_default_does_not_write_and_json_passes():
    receipt_path = ROOT / "docs" / "P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json"
    before = receipt_path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "radjax_student.learning.p3_5_acceptance", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "pass"
    assert receipt_path.read_bytes() == before
