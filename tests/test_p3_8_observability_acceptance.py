from dataclasses import replace

import pytest

from radjax_student.learning import LearningIssue
from radjax_student.learning.hooks import HookDispatchResult
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
    _default_dependencies,
    _has_test_placeholder,
    _source_has_forbidden_import,
    main,
    run_p3_8_observability_acceptance,
)
from radjax_student.learning.telemetry import MetricSeries


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
    assert _audit_metrics(_default_dependencies())


def test_13_hook_audit_passes():
    assert _audit_hooks(_default_dependencies())


def test_14_loop_integration_audit_passes():
    assert _audit_loop_integration(_default_dependencies())


def test_15_run_reporting_audit_passes():
    assert _audit_run_reporting(_default_dependencies())


def test_16_deterministic_replay_audit_passes():
    assert _audit_deterministic_replay(_default_dependencies())


def test_17_failure_path_audit_passes():
    assert _audit_failure_paths(_default_dependencies())


def test_18_observer_only_audit_passes():
    assert _audit_observer_only_boundary(_default_dependencies())


def test_19_bounded_history_audit_passes():
    assert _audit_bounded_history(_default_dependencies())


def test_20_import_boundary_audit_passes():
    assert _audit_import_boundary(_default_dependencies())


def test_21_documentation_audit_passes():
    assert _audit_documentation(_default_dependencies())


def test_22_test_inventory_audit_passes():
    assert _audit_test_inventory(_default_dependencies())


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


def dependencies(**changes):
    return replace(_default_dependencies(), **changes)


def codes(receipt):
    return [blocker.code for blocker in receipt.blockers]


class BrokenRetentionSeries(MetricSeries):
    @property
    def records(self):
        return tuple(self._all)


class BrokenSummarySeries(MetricSeries):
    def summary(self):
        return replace(super().summary(), mean=-1.0)


def test_41_broken_metric_retention_detected():
    result = run_p3_8_observability_acceptance(
        dependencies(metric_series_factory=BrokenRetentionSeries)
    )
    assert (
        not result.metrics_contract_valid
        and "p3_8_metrics_contract_failed" in codes(result)
    )


def test_42_broken_metric_summary_detected():
    result = run_p3_8_observability_acceptance(
        dependencies(metric_series_factory=BrokenSummarySeries)
    )
    assert (
        not result.metrics_contract_valid
        and "p3_8_metrics_contract_failed" in codes(result)
    )


def test_43_reversed_hook_ordering_detected():
    base = _default_dependencies().dispatch_hooks_fn

    def reversed_dispatch(*args, **kwargs):
        result = base(*args, **kwargs)
        return replace(result, receipts=tuple(reversed(result.receipts)))

    result = run_p3_8_observability_acceptance(
        dependencies(dispatch_hooks_fn=reversed_dispatch)
    )
    assert not result.hook_contract_valid and "p3_8_hook_contract_failed" in codes(
        result
    )


def test_44_lost_disable_failure_class_detected():
    base = _default_dependencies().dispatch_hooks_fn

    def stripped_dispatch(*args, **kwargs):
        result = base(*args, **kwargs)
        if result.disabled_hook_ids:
            return HookDispatchResult(
                "warning",
                result.receipts,
                result.metrics,
                (LearningIssue("learning_hook_disabled", "lost class"),),
                (),
                result.disabled_hook_ids,
            )
        return result

    result = run_p3_8_observability_acceptance(
        dependencies(dispatch_hooks_fn=stripped_dispatch)
    )
    assert not result.hook_contract_valid and "p3_8_hook_contract_failed" in codes(
        result
    )


def test_45_missing_exception_type_detected():
    base = _default_dependencies().dispatch_hooks_fn

    def stripped_dispatch(*args, **kwargs):
        result = base(*args, **kwargs)
        return replace(
            result,
            receipts=tuple(
                replace(receipt, metadata={}) for receipt in result.receipts
            ),
        )

    result = run_p3_8_observability_acceptance(
        dependencies(dispatch_hooks_fn=stripped_dispatch)
    )
    assert not result.hook_contract_valid and "p3_8_hook_contract_failed" in codes(
        result
    )


def test_46_missing_lifecycle_event_detected():
    base = _default_dependencies().run_loop_fn

    def missing_event(**kwargs):
        result = base(**kwargs)
        return replace(
            result,
            hook_events=tuple(
                event for event in result.hook_events if event != "step_end"
            ),
        )

    result = run_p3_8_observability_acceptance(dependencies(run_loop_fn=missing_event))
    assert (
        not result.loop_integration_valid
        and "p3_8_loop_integration_failed" in codes(result)
    )


def test_47_lost_hook_blocker_detected():
    base = _default_dependencies().run_loop_fn

    def lost_blocker(**kwargs):
        result = base(**kwargs)
        if result.stop_reason == "hook_failure":
            return replace(result, hook_blockers=())
        return result

    result = run_p3_8_observability_acceptance(dependencies(run_loop_fn=lost_blocker))
    assert (
        not result.loop_integration_valid
        and "p3_8_loop_integration_failed" in codes(result)
    )


def test_48_lost_hook_metric_flow_detected():
    base = _default_dependencies().run_loop_fn

    def lost_metric(**kwargs):
        result = base(**kwargs)
        return replace(
            result,
            metrics=tuple(
                metric for metric in result.metrics if metric.name != "hook.metric"
            ),
        )

    result = run_p3_8_observability_acceptance(dependencies(run_loop_fn=lost_metric))
    assert (
        not result.loop_integration_valid
        and "p3_8_loop_integration_failed" in codes(result)
    )


def test_49_lost_hook_warning_flow_detected():
    base = _default_dependencies().run_loop_fn

    def lost_warning(**kwargs):
        result = base(**kwargs)
        return replace(
            result,
            warnings=tuple(
                warning for warning in result.warnings if warning.code != "hook.warning"
            ),
        )

    result = run_p3_8_observability_acceptance(dependencies(run_loop_fn=lost_warning))
    assert (
        not result.loop_integration_valid
        and "p3_8_loop_integration_failed" in codes(result)
    )


def test_50_fabricated_global_step_detected():
    base = _default_dependencies().build_report_fn

    def fabricated_step(**kwargs):
        report = base(**kwargs)
        return replace(
            report,
            status=replace(report.status, global_step=report.status.steps_completed),
        )

    result = run_p3_8_observability_acceptance(
        dependencies(build_report_fn=fabricated_step)
    )
    assert not result.run_reporting_valid and "p3_8_run_reporting_failed" in codes(
        result
    )


def test_51_nondeterministic_json_detected():
    base = _default_dependencies().build_report_fn
    calls = []

    def unstable_report(**kwargs):
        report = base(**kwargs)
        calls.append(report)
        return replace(report, metadata={"sequence": len(calls)})

    result = run_p3_8_observability_acceptance(
        dependencies(build_report_fn=unstable_report)
    )
    assert not result.run_reporting_valid and "p3_8_run_reporting_failed" in codes(
        result
    )


def test_52_nondeterministic_loop_replay_detected():
    base = _default_dependencies().run_loop_fn
    calls = []

    def unstable_loop(**kwargs):
        result = base(**kwargs)
        calls.append(result)
        return replace(result, global_step=result.global_step + len(calls) % 2)

    result = run_p3_8_observability_acceptance(dependencies(run_loop_fn=unstable_loop))
    assert (
        not result.deterministic_replay_valid
        and "p3_8_deterministic_replay_failed" in codes(result)
    )


def test_53_overwritten_core_failure_reason_detected():
    base = _default_dependencies().run_loop_fn

    def overwritten_reason(**kwargs):
        result = base(**kwargs)
        if result.stop_reason == "learning_step_failure":
            return replace(result, stop_reason="hook_failure")
        return result

    result = run_p3_8_observability_acceptance(
        dependencies(run_loop_fn=overwritten_reason)
    )
    assert not result.failure_paths_valid and "p3_8_failure_paths_failed" in codes(
        result
    )


def test_54_observer_mutation_detected():
    base = _default_dependencies().build_report_fn

    def mutating_builder(**kwargs):
        kwargs["metadata"]["sentinel"] = "mutated"
        return base(**kwargs)

    result = run_p3_8_observability_acceptance(
        dependencies(build_report_fn=mutating_builder)
    )
    assert (
        not result.observer_only_boundary_valid
        and "p3_8_observer_boundary_failed" in codes(result)
    )


def test_55_false_complete_history_claim_detected():
    base = _default_dependencies().build_report_fn

    def complete_history_builder(**kwargs):
        report = base(**kwargs)
        return replace(report, metadata={"metric_summary_source": "complete_history"})

    result = run_p3_8_observability_acceptance(
        dependencies(build_report_fn=complete_history_builder)
    )
    assert (
        not result.bounded_history_claim_valid
        and "p3_8_bounded_history_claim_failed" in codes(result)
    )


def test_56_import_torch_alias_detected():
    assert _source_has_forbidden_import("import torch as t", ("torch",))


def test_57_importlib_wandb_detected():
    assert _source_has_forbidden_import('importlib.import_module("wandb")', ("wandb",))


def test_58_dunder_import_requests_detected():
    assert _source_has_forbidden_import('__import__("requests")', ("requests",))


def test_59_missing_documentation_detected():
    base = _default_dependencies().path_exists_fn
    result = run_p3_8_observability_acceptance(
        dependencies(
            path_exists_fn=lambda path: (
                False if path.name.startswith("P3_8D") else base(path)
            )
        )
    )
    assert not result.documentation_valid and "p3_8_documentation_failed" in codes(
        result
    )


def test_60_literal_assertion_detected():
    assert _has_test_placeholder("def test_example():\n    assert " + "True\n")


def test_61_skip_call_detected():
    assert _has_test_placeholder("def test_example():\n    pytest" + ".skip('x')\n")


def test_62_imported_test_function_detected():
    assert _has_test_placeholder(
        "from "
        + "tests.other import "
        + "test_other\ndef test_example():\n    return 1\n"
    )


def test_63_called_test_function_detected():
    assert _has_test_placeholder("def test_example():\n    test_other()\n")


def test_64_pass_in_test_detected():
    assert _has_test_placeholder("def test_example():\n    " + "pass\n")


def test_65_unexpected_audit_exception_returns_fail_receipt():
    result = run_p3_8_observability_acceptance(
        dependencies(
            metric_series_factory=lambda *args: (_ for _ in ()).throw(RuntimeError())
        )
    )
    assert result.status == "fail" and not result.metrics_contract_valid


def test_66_unexpected_audit_exception_adds_internal_error():
    result = run_p3_8_observability_acceptance(
        dependencies(
            metric_series_factory=lambda *args: (_ for _ in ()).throw(RuntimeError())
        )
    )
    assert "p3_8_internal_error" in codes(result)


def test_67_acceptance_command_returns_nonzero_on_failed_receipt():
    failed = dependencies(
        metric_series_factory=lambda *args: (_ for _ in ()).throw(RuntimeError())
    )
    assert main((), failed) == 1
