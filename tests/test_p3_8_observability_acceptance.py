import pytest

from radjax_student.learning import LearningIssue
from radjax_student.learning.observability_acceptance import (
    CLAIMS,
    NON_CLAIMS,
    SCHEMA,
    P38ObservabilityAcceptanceReceipt,
    _audit_bounded_history,
    _audit_deterministic_replay,
    _audit_documentation,
    _audit_failure_paths,
    _audit_hooks,
    _audit_import_boundary,
    _audit_loop_integration,
    _audit_metrics,
    _audit_observer_only_boundary,
    _audit_run_reporting,
    _audit_test_inventory,
    _receipt_from_values,
    run_p3_8_observability_acceptance,
)


def values(**changes):
    result = {
        "metrics_contract_valid": True,
        "hook_contract_valid": True,
        "loop_integration_valid": True,
        "run_reporting_valid": True,
        "deterministic_replay_valid": True,
        "failure_paths_valid": True,
        "observer_only_boundary_valid": True,
        "bounded_history_claim_valid": True,
        "import_boundary_valid": True,
        "documentation_valid": True,
        "test_inventory_valid": True,
    }
    result.update(changes)
    return result


def receipt(**changes):
    payload = values(**changes)
    status = "pass" if all(payload.values()) else "fail"
    return P38ObservabilityAcceptanceReceipt(
        schema_version=SCHEMA,
        status=status,
        blockers=(),
        warnings=(),
        metadata={"test": "p3_8d"},
        **payload,
    )


def test_01_valid_passing_receipt():
    assert receipt().status == "pass"


def test_02_pass_rejected_when_flag_false():
    with pytest.raises(ValueError):
        P38ObservabilityAcceptanceReceipt(
            schema_version=SCHEMA, status="pass", **values(metrics_contract_valid=False)
        )


def test_03_pass_rejected_with_blocker():
    with pytest.raises(ValueError):
        P38ObservabilityAcceptanceReceipt(
            schema_version=SCHEMA,
            status="pass",
            blockers=(LearningIssue("blocked", "blocked"),),
            **values(),
        )


def test_04_fail_rejected_when_everything_passes():
    with pytest.raises(ValueError):
        P38ObservabilityAcceptanceReceipt(
            schema_version=SCHEMA, status="fail", **values()
        )


def test_05_invalid_schema_rejected():
    with pytest.raises(ValueError):
        P38ObservabilityAcceptanceReceipt(
            schema_version="wrong", status="pass", **values()
        )


def test_06_non_boolean_flag_rejected():
    with pytest.raises(TypeError):
        P38ObservabilityAcceptanceReceipt(
            schema_version=SCHEMA, status="pass", **values(metrics_contract_valid=1)
        )


def test_07_invalid_blocker_rejected():
    with pytest.raises(TypeError):
        P38ObservabilityAcceptanceReceipt(
            schema_version=SCHEMA,
            status="fail",
            blockers=(object(),),
            **values(metrics_contract_valid=False),
        )


def test_08_duplicate_claim_rejected():
    with pytest.raises(ValueError):
        P38ObservabilityAcceptanceReceipt(
            schema_version=SCHEMA,
            status="pass",
            claims_made=CLAIMS + (CLAIMS[0],),
            **values(),
        )


def test_09_invalid_metadata_rejected():
    with pytest.raises(ValueError):
        P38ObservabilityAcceptanceReceipt(
            schema_version=SCHEMA,
            status="pass",
            metadata={"parameters": {}},
            **values(),
        )


def test_10_receipt_dict_is_deterministic():
    assert receipt().to_dict() == receipt().to_dict()


def test_11_receipt_json_is_deterministic():
    assert receipt().to_json() == receipt().to_json()


def test_12_metrics_audit_passes():
    assert _audit_metrics()


def test_13_hook_audit_passes():
    assert _audit_hooks()


def test_14_loop_integration_audit_passes():
    assert _audit_loop_integration()


def test_15_run_reporting_audit_passes():
    assert _audit_run_reporting()


def test_16_deterministic_replay_audit_passes():
    assert _audit_deterministic_replay()


def test_17_failure_path_audit_passes():
    assert _audit_failure_paths()


def test_18_observer_only_audit_passes():
    assert _audit_observer_only_boundary()


def test_19_bounded_history_audit_passes():
    assert _audit_bounded_history()


def test_20_import_boundary_audit_passes():
    assert _audit_import_boundary()


def test_21_documentation_audit_passes():
    assert _audit_documentation()


def test_22_test_inventory_audit_passes():
    assert _audit_test_inventory()


def test_23_tampered_metric_behavior_fails_receipt():
    assert _receipt_from_values(values(metrics_contract_valid=False)).status == "fail"


def test_24_tampered_hook_ordering_fails_receipt():
    assert _receipt_from_values(values(hook_contract_valid=False)).status == "fail"


def test_25_missing_lifecycle_event_fails_receipt():
    assert _receipt_from_values(values(loop_integration_valid=False)).status == "fail"


def test_26_lost_hook_blocker_fails_receipt():
    assert _receipt_from_values(values(failure_paths_valid=False)).status == "fail"


def test_27_fabricated_global_step_fails_receipt():
    assert _receipt_from_values(values(run_reporting_valid=False)).status == "fail"


def test_28_nondeterministic_report_fails_receipt():
    assert (
        _receipt_from_values(values(deterministic_replay_valid=False)).status == "fail"
    )


def test_29_forbidden_import_fails_receipt():
    assert _receipt_from_values(values(import_boundary_valid=False)).status == "fail"


def test_30_missing_documentation_fails_receipt():
    assert _receipt_from_values(values(documentation_valid=False)).status == "fail"


def test_31_placeholder_pattern_fails_receipt():
    assert _receipt_from_values(values(test_inventory_valid=False)).status == "fail"


def test_32_full_acceptance_returns_pass():
    assert run_p3_8_observability_acceptance().status == "pass"


def test_33_all_validity_flags_are_true():
    assert all(
        value
        for key, value in run_p3_8_observability_acceptance().to_dict().items()
        if key.endswith("_valid")
    )


def test_34_full_acceptance_has_no_blockers():
    assert run_p3_8_observability_acceptance().blockers == ()


def test_35_claims_are_exact():
    assert run_p3_8_observability_acceptance().claims_made == CLAIMS


def test_36_required_non_claims_are_present():
    assert set(NON_CLAIMS).issubset(run_p3_8_observability_acceptance().claims_not_made)


def test_37_two_runs_have_identical_dicts():
    assert (
        run_p3_8_observability_acceptance().to_dict()
        == run_p3_8_observability_acceptance().to_dict()
    )


def test_38_two_runs_have_identical_json():
    assert (
        run_p3_8_observability_acceptance().to_json()
        == run_p3_8_observability_acceptance().to_json()
    )


def test_39_receipt_contains_no_forbidden_state():
    payload = run_p3_8_observability_acceptance().to_json()
    assert all(
        name not in payload for name in ("parameters", "optimizer_state", "raw_batch")
    )


def test_40_acceptance_uses_no_external_telemetry_or_network():
    payload = run_p3_8_observability_acceptance().to_json()
    assert "external_telemetry" in payload and "network" not in payload
