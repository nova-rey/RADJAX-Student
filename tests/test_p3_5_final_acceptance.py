from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.learning import LearningIssue
from radjax_student.learning.p3_5_acceptance import (
    FLAGS,
    SCHEMA,
    GateCheck,
    P35AcceptanceDependencies,
    P35ArchitectureIntegrityReceipt,
    run_p3_5_architecture_integrity_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]


def _issue(code: str = "learning_step_failed") -> LearningIssue:
    return LearningIssue.create(code, "injected adversarial failure")


def _pass(**evidence):
    return GateCheck(True, evidence)


def _fail(**evidence):
    return GateCheck(False, evidence, _issue())


def _audit(_: Path):
    return {"status": "pass", "blockers": [], "module_count": 1}


def _checks():
    return _pass(registry="ArchitectureRegistry"), _pass(explicit_executor=True)


def _command(_: Path, target: str):
    return _pass(target=target, returncode=0)


def _passing_dependencies(**changes):
    deps = P35AcceptanceDependencies(
        build_audit=_audit,
        run_jax_contract=lambda: _pass(contract="jax"),
        check_namespace_legacy=_checks,
        check_hf=lambda: _pass(mapping_count=1),
        check_checkpoint=lambda: _pass(codec="json"),
        check_docs=lambda _: _pass(document="p35"),
        run_phase1=_command,
        run_phase2=_command,
        run_phase3=_command,
        check_import_purity=lambda _: _pass(modules=("base",)),
    )
    return dataclasses.replace(deps, **changes)


def _receipt(deps=None):
    return run_p3_5_architecture_integrity_acceptance(
        ROOT, dependencies=deps or _passing_dependencies()
    )


def _assert_failure(flag: str, deps):
    receipt = _receipt(deps)
    assert receipt.status == "fail"
    assert not getattr(receipt, flag)
    assert receipt.blockers and isinstance(receipt.blockers[0], LearningIssue)


def _valid_values(**changes):
    values = {name: True for name in FLAGS}
    values.update(changes)
    return values


def test_01_valid_passing_receipt_is_machine_readable():
    receipt = _receipt()
    assert receipt.status == "pass" and receipt.schema_version == SCHEMA


def test_02_receipt_schema_is_exact():
    with pytest.raises(ValueError, match="schema"):
        P35ArchitectureIntegrityReceipt("wrong", "pass", **_valid_values())


def test_03_receipt_flags_must_be_actual_booleans():
    with pytest.raises(TypeError, match="booleans"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "pass", **_valid_values(import_purity_valid=1)
        )


def test_04_receipt_rejects_pass_with_false_flag():
    with pytest.raises(ValueError, match="status"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "pass", **_valid_values(phase1_regression_valid=False)
        )


def test_05_receipt_rejects_pass_with_blocker():
    with pytest.raises(ValueError, match="status"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "pass", blockers=(_issue(),), **_valid_values()
        )


def test_06_receipt_rejects_fail_without_evidence():
    with pytest.raises(ValueError, match="requires blocker"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA, "fail", **_valid_values(phase1_regression_valid=False)
        )


def test_07_receipt_rejects_invalid_finding():
    with pytest.raises(TypeError, match="LearningIssue"):
        P35ArchitectureIntegrityReceipt(
            SCHEMA,
            "fail",
            blockers=("bad",),
            **_valid_values(phase1_regression_valid=False),
        )


def test_08_receipt_metadata_is_immutable():
    with pytest.raises(TypeError):
        _receipt().metadata["tamper"] = True


def test_09_receipt_dict_is_deterministic():
    assert _receipt().to_dict() == _receipt().to_dict()


def test_10_receipt_json_is_deterministic():
    assert _receipt().to_json() == _receipt().to_json()


def test_11_receipt_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        P35ArchitectureIntegrityReceipt(SCHEMA, "unknown", **_valid_values())


def test_12_internal_cycle_fails_full_gate():
    def cycle(_: Path):
        return {
            "status": "blocked",
            "blockers": [{"code": "dependency_cycle"}],
            "module_count": 1,
        }

    _assert_failure(
        "dependency_boundaries_valid", _passing_dependencies(build_audit=cycle)
    )


def test_13_architecture_runtime_dependency_violation_fails_full_gate():
    def violation(_: Path):
        return {
            "status": "blocked",
            "blockers": [{"code": "architecture_imports_runtime"}],
            "module_count": 1,
        }

    _assert_failure(
        "dependency_boundaries_valid", _passing_dependencies(build_audit=violation)
    )


def test_14_runtime_concrete_architecture_violation_fails_full_gate():
    def violation(_: Path):
        return {
            "status": "blocked",
            "blockers": [{"code": "runtime_imports_architecture"}],
            "module_count": 1,
        }

    _assert_failure(
        "dependency_boundaries_valid", _passing_dependencies(build_audit=violation)
    )


def test_15_core_legacy_import_and_scripts_import_fail_full_gate():
    for code in ("core_imports_legacy", "installed_package_imports_scripts"):

        def violation(_: Path, code=code):
            return {
                "status": "blocked",
                "blockers": [{"code": code}],
                "module_count": 1,
            }

        _assert_failure(
            "dependency_boundaries_valid", _passing_dependencies(build_audit=violation)
        )


def test_16_dense_export_and_duplicate_registry_fail_full_gate():
    for code in ("dense_targets_public_export", "duplicate_architecture_registry"):

        def violation(_: Path, code=code):
            return {
                "status": "blocked",
                "blockers": [{"code": code}],
                "module_count": 1,
            }

        _assert_failure(
            "dependency_boundaries_valid", _passing_dependencies(build_audit=violation)
        )


def test_17_objective_parameter_access_and_ceremonial_forward_fail_full_gate():
    for code in ("objective_receives_raw_parameters", "forward_result_discarded"):

        def violation(_: Path, code=code):
            return {
                "status": "blocked",
                "blockers": [{"code": code}],
                "module_count": 1,
            }

        _assert_failure(
            "architecture_objective_separation_valid",
            _passing_dependencies(build_audit=violation),
        )


def test_18_duplicate_forward_missing_surface_and_string_scope_fail_full_gate():
    for code in (
        "duplicated_forward_application",
        "missing_objective_surface",
        "string_objective_scope",
    ):

        def violation(_: Path, code=code):
            return {
                "status": "blocked",
                "blockers": [{"code": code}],
                "module_count": 1,
            }

        _assert_failure(
            "architecture_objective_separation_valid",
            _passing_dependencies(build_audit=violation),
        )


def test_19_missing_jax_capability_fails_without_fallback():
    _assert_failure(
        "jax_native_learning_contract_valid",
        _passing_dependencies(run_jax_contract=_fail),
    )


def test_20_numpy_jit_and_device_leaks_fail_full_gate():
    for evidence in ("numpy_import", "direct_jit", "device_selection"):

        def jax_failure(evidence=evidence):
            return _fail(regression=evidence)

        _assert_failure(
            "jax_native_learning_contract_valid",
            _passing_dependencies(run_jax_contract=jax_failure),
        )


def test_21_nonfinite_and_differentiated_or_mutated_carry_fail_full_gate():
    for evidence in (
        "nonfinite_loss",
        "nonfinite_gradient",
        "differentiated_carry",
        "mutated_carry",
    ):

        def jax_failure(evidence=evidence):
            return _fail(regression=evidence)

        _assert_failure(
            "jax_native_learning_contract_valid",
            _passing_dependencies(run_jax_contract=jax_failure),
        )


def test_22_replay_excluded_parameter_and_runtime_bypass_fail_full_gate():
    for evidence in (
        "replay_divergence",
        "excluded_parameter_mutation",
        "runtime_jit_bypass",
    ):

        def jax_failure(evidence=evidence):
            return _fail(regression=evidence)

        _assert_failure(
            "jax_native_learning_contract_valid",
            _passing_dependencies(run_jax_contract=jax_failure),
        )


def test_23_students_and_scalar_root_exports_fail_full_gate():
    _assert_failure(
        "architecture_namespace_valid",
        _passing_dependencies(
            check_namespace_legacy=lambda: (
                _fail(regression="students_export"),
                _pass(),
            )
        ),
    )
    _assert_failure(
        "legacy_isolation_valid",
        _passing_dependencies(
            check_namespace_legacy=lambda: (_pass(), _fail(regression="scalar_export"))
        ),
    )


def test_24_loop_default_shim_warning_and_jax_legacy_fallback_fail_full_gate():
    _assert_failure(
        "legacy_isolation_valid",
        _passing_dependencies(
            check_namespace_legacy=lambda: (_pass(), _fail(regression="loop_default"))
        ),
    )
    _assert_failure(
        "legacy_isolation_valid",
        _passing_dependencies(
            check_namespace_legacy=lambda: (_pass(), _fail(regression="shim_warning"))
        ),
    )
    _assert_failure(
        "jax_native_learning_contract_valid",
        _passing_dependencies(
            run_jax_contract=lambda: _fail(regression="legacy_fallback")
        ),
    )


def test_25_hf_empty_and_duplicate_mappings_fail_full_gate():
    for evidence in (
        "empty_mapping",
        "duplicate_logical",
        "duplicate_jax",
        "duplicate_hf",
    ):

        def broken_hf(evidence=evidence):
            return _fail(regression=evidence)

        _assert_failure(
            "hf_preservation_contract_valid", _passing_dependencies(check_hf=broken_hf)
        )


def test_26_hf_catalog_shape_runtime_and_config_tampers_fail_full_gate():
    for evidence in (
        "missing_catalog",
        "shape_mismatch",
        "runtime_key",
        "config_conflict",
        "transformers_import",
    ):

        def broken_hf(evidence=evidence):
            return _fail(regression=evidence)

        _assert_failure(
            "hf_preservation_contract_valid", _passing_dependencies(check_hf=broken_hf)
        )


def test_27_checkpoint_role_and_descriptor_tampers_fail_full_gate():
    for evidence in (
        "role_conflation",
        "pytree_v2_claim",
        "tensor_emitted",
        "runtime_handle",
        "invalid_descriptor",
    ):

        def broken_checkpoint(evidence=evidence):
            return _fail(regression=evidence)

        _assert_failure(
            "checkpoint_ownership_valid",
            _passing_dependencies(check_checkpoint=broken_checkpoint),
        )


def test_28_phase1_command_failure_cannot_be_overridden_by_json():
    _assert_failure(
        "phase1_regression_valid",
        _passing_dependencies(
            run_phase1=lambda _, target: _fail(target=target, returncode=1)
        ),
    )


def test_29_phase2_command_failure_fails_full_gate():
    _assert_failure(
        "phase2_regression_valid",
        _passing_dependencies(
            run_phase2=lambda _, target: _fail(target=target, returncode=1)
        ),
    )


def test_30_phase3_failure_and_stale_receipt_fail_full_gate():
    _assert_failure(
        "phase3_regression_valid",
        _passing_dependencies(
            run_phase3=lambda _, target: _fail(
                target=target, returncode=1, stale_receipt="pass"
            )
        ),
    )


def test_31_import_probe_jax_torch_and_tome_leaks_fail_full_gate():
    for leaked in ("jax", "torch", "radjax_tome"):

        def impure(_: Path, leaked=leaked):
            return _fail(leaked=leaked)

        _assert_failure(
            "import_purity_valid", _passing_dependencies(check_import_purity=impure)
        )


@pytest.mark.jax
def test_32_real_gate_passes_in_jax_environment():
    assert run_p3_5_architecture_integrity_acceptance(ROOT).status == "pass"


@pytest.mark.jax
def test_33_recorded_receipt_matches_real_gate():
    recorded = json.loads(
        (ROOT / "docs" / "P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json").read_text()
    )
    assert recorded == run_p3_5_architecture_integrity_acceptance(ROOT).to_dict()


@pytest.mark.jax
def test_34_cli_human_and_json_pass_without_default_write():
    receipt_path = ROOT / "docs" / "P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json"
    before = receipt_path.read_bytes()
    human = subprocess.run(
        [sys.executable, "-m", "radjax_student.learning.p3_5_acceptance"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    encoded = subprocess.run(
        [sys.executable, "-m", "radjax_student.learning.p3_5_acceptance", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert human.returncode == encoded.returncode == 0
    assert "Architecture Integrity: pass" in human.stdout
    assert json.loads(encoded.stdout)["status"] == "pass"
    assert receipt_path.read_bytes() == before


def test_35_injected_failure_and_internal_error_fail_closed():
    _assert_failure(
        "documentation_consistency_valid",
        _passing_dependencies(check_docs=lambda _: _fail(regression="stale_docs")),
    )

    def boom(_: Path):
        raise RuntimeError("unexpected")

    receipt = _receipt(_passing_dependencies(build_audit=boom))
    assert (
        receipt.status == "fail"
        and receipt.blockers[0].code == "learning_internal_error"
    )


def test_36_cli_write_receipt_is_explicit_only(tmp_path):
    target = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "radjax_student.learning.p3_5_acceptance",
            "--write-receipt",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in {0, 1}
    assert target.is_file()
