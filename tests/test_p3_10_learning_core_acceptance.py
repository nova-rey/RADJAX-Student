from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import pytest

from radjax_student.learning import LearningIssue
from radjax_student.learning.p3_10_acceptance import (
    CLAIMS,
    NON_CLAIMS,
    SCHEMA,
    VALIDITY_FIELDS,
    P310LearningCoreAcceptanceReceipt,
    _default_dependencies,
    _default_documentation,
    _default_inventory,
    _default_replay,
    _default_synthetic,
    main,
    run_p3_10_learning_core_acceptance,
)


@lru_cache(maxsize=1)
def receipt():
    return run_p3_10_learning_core_acceptance()


def codes(value):
    return {issue.code for issue in value.blockers}


def valid_receipt(**changes):
    values = {
        "schema_version": SCHEMA,
        "status": "pass",
        **{name: True for name in VALIDITY_FIELDS},
    }
    values.update(changes)
    return P310LearningCoreAcceptanceReceipt(**values)


def test_01_gate_passes():
    assert receipt().status == "pass"


def test_02_schema_is_exact():
    assert receipt().schema_version == SCHEMA


def test_03_all_validity_flags_pass():
    assert all(getattr(receipt(), name) for name in VALIDITY_FIELDS)


def test_04_receipt_has_no_blockers():
    assert receipt().blockers == ()


def test_05_receipt_is_immutable():
    with pytest.raises((AttributeError, TypeError)):
        receipt().status = "fail"


def test_06_receipt_dict_is_deterministic():
    assert receipt().to_dict() == receipt().to_dict()


def test_07_receipt_json_is_deterministic():
    assert receipt().to_json() == receipt().to_json()


def test_08_claims_are_exact():
    assert receipt().claims_made == CLAIMS


def test_09_required_nonclaims_are_present():
    assert set(NON_CLAIMS).issubset(receipt().claims_not_made)


def test_10_pass_requires_all_flags():
    with pytest.raises(ValueError):
        valid_receipt(contracts_valid=False)


def test_11_pass_rejects_blocker():
    with pytest.raises(ValueError):
        valid_receipt(blockers=(LearningIssue("blocked", "blocked"),))


def test_12_fail_requires_evidence():
    with pytest.raises(ValueError):
        valid_receipt(status="fail")


def test_13_invalid_schema_rejected():
    with pytest.raises(ValueError):
        valid_receipt(schema_version="wrong")


def test_14_nonboolean_flag_rejected():
    with pytest.raises(TypeError):
        valid_receipt(loop_valid=1)


def test_15_invalid_issue_rejected():
    with pytest.raises(TypeError):
        valid_receipt(blockers=(object(),))


def test_16_duplicate_claim_rejected():
    with pytest.raises(ValueError):
        valid_receipt(claims_made=CLAIMS + (CLAIMS[0],))


def test_17_contracts_section_passes():
    assert receipt().contracts_valid


def test_18_optimizer_section_passes():
    assert receipt().optimizer_valid


def test_19_single_step_section_passes():
    assert receipt().single_step_valid


def test_20_loop_section_passes():
    assert receipt().loop_valid


def test_21_checkpoint_section_passes():
    assert receipt().checkpoint_valid


def test_22_resume_section_passes():
    assert receipt().resume_valid


def test_23_observability_section_passes():
    assert receipt().observability_valid


def test_24_synthetic_section_passes():
    assert receipt().synthetic_learning_valid


def test_25_replay_section_passes():
    assert receipt().deterministic_replay_valid


def test_26_documentation_section_passes():
    assert receipt().documentation_valid


def test_27_inventory_section_passes():
    assert receipt().test_inventory_valid


def test_28_contract_tamper_is_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), contracts_fn=lambda: {})
    )
    assert result.status == "fail" and "p3_10_contracts_failed" in codes(result)


def test_29_optimizer_tamper_is_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            optimizer_fn=lambda: {
                "optimizer_id": "wrong",
                "step": 1,
                "changed": (),
                "parameters": {},
            },
        )
    )
    assert result.status == "fail" and "p3_10_optimizer_failed" in codes(result)


def test_30_single_step_tamper_is_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            single_step_fn=lambda: {"status": "fail", "loss_decrease": False},
        )
    )
    assert result.status == "fail" and "p3_10_single_step_failed" in codes(result)


def test_31_loop_tamper_is_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            loop_fn=lambda: {"status": "pass", "steps": 11, "stop_reason": "max_steps"},
        )
    )
    assert result.status == "fail" and "p3_10_loop_failed" in codes(result)


def test_32_checkpoint_tamper_is_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_fn=lambda: {
                "schema": "wrong",
                "source_state": {"position": 1},
                "source_owned": True,
                "source_hashed": True,
                "source_sized": True,
            },
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)


def test_33_resume_tamper_is_detected():
    bad = replace(
        _default_synthetic(),
        status="fail",
        blockers=(LearningIssue("resume", "resume"),),
        checkpoint_restore_valid=False,
    )
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), resume_fn=lambda: bad)
    )
    assert result.status == "fail" and "p3_10_resume_failed" in codes(result)


def test_34_observability_tamper_is_detected():
    class FailedObservation:
        status = "fail"

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), observability_fn=lambda: FailedObservation())
    )
    assert result.status == "fail" and "p3_10_observability_failed" in codes(result)


def test_35_synthetic_tamper_is_detected():
    bad = replace(
        _default_synthetic(),
        status="fail",
        blockers=(LearningIssue("synthetic", "synthetic"),),
    )
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), synthetic_fn=lambda: bad)
    )
    assert result.status == "fail" and "p3_10_synthetic_learning_failed" in codes(
        result
    )


def test_36_replay_tamper_is_detected():
    first, second = _default_replay()
    bad = replace(second, status="fail", blockers=(LearningIssue("replay", "replay"),))
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), replay_fn=lambda: (first, bad))
    )
    assert result.status == "fail" and "p3_10_deterministic_replay_failed" in codes(
        result
    )


def test_37_documentation_tamper_is_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), documentation_fn=lambda: {"p36_v2": False})
    )
    assert result.status == "fail" and "p3_10_documentation_failed" in codes(result)


def test_38_inventory_tamper_is_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            test_inventory_fn=lambda: {"named_tests": 1, "required": 50, "files": 3},
        )
    )
    assert result.status == "fail" and "p3_10_test_inventory_failed" in codes(result)


def test_39_unexpected_contract_exception_fails_closed():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            contracts_fn=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
        )
    )
    assert result.status == "fail" and "p3_10_internal_error" in codes(result)


def test_40_unexpected_optimizer_exception_fails_closed():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            optimizer_fn=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
        )
    )
    assert result.status == "fail" and "p3_10_internal_error" in codes(result)


def test_41_unexpected_synthetic_exception_fails_closed():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            synthetic_fn=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
        )
    )
    assert result.status == "fail" and "p3_10_internal_error" in codes(result)


def test_42_default_documentation_evidence_is_real():
    evidence = _default_documentation()
    assert all(evidence.values())


def test_43_default_inventory_evidence_is_real():
    evidence = _default_inventory()
    assert evidence["named_tests"] >= 50


def test_44_gate_json_contains_all_flags():
    payload = receipt().to_dict()
    assert all(name in payload for name in VALIDITY_FIELDS)


def test_45_gate_json_is_sorted_and_stable():
    assert receipt().to_json() == receipt().to_json()


def test_46_gate_claims_are_unique():
    assert len(set(receipt().claims_made)) == len(receipt().claims_made)


def test_47_gate_nonclaims_include_quality_limitations():
    assert {"model_quality", "evaluation", "generalization"}.issubset(
        receipt().claims_not_made
    )


def test_48_gate_metadata_identifies_p310():
    assert receipt().metadata["gate"] == "P3.10"


def test_49_gate_section_count_is_exact():
    assert receipt().metadata["section_count"] == len(VALIDITY_FIELDS)


def test_50_cli_passes():
    assert main([]) == 0


def test_51_cli_json_passes():
    assert main(["--json"]) == 0


def test_52_cli_fails_for_injected_contract_error():
    deps = replace(_default_dependencies(), contracts_fn=lambda: {})
    assert run_p3_10_learning_core_acceptance(deps).status == "fail"


def test_53_public_lazy_export_resolves():
    from radjax_student.learning import (
        P310LearningCoreAcceptanceReceipt,
        run_p3_10_learning_core_acceptance,
    )

    assert P310LearningCoreAcceptanceReceipt is not None
    assert run_p3_10_learning_core_acceptance().status == "pass"


def test_54_public_acceptance_is_repeatable():
    assert run_p3_10_learning_core_acceptance().to_dict() == receipt().to_dict()


def test_55_acceptance_has_no_traceback_in_json():
    assert "traceback" not in receipt().to_json().lower()
