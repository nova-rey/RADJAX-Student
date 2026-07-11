from dataclasses import replace
from types import SimpleNamespace

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


def test_51a_merged_warning_and_blocker_reporting_detected():
    base = _default_dependencies().build_report_fn

    class MergedReport:
        def __init__(self, report):
            self._report = report

        @property
        def issues(self):
            return SimpleNamespace(
                warning_codes=self._report.issues.warning_codes
                + self._report.issues.hook_blocker_codes,
                hook_blocker_codes=(),
            )

        def __getattr__(self, name):
            return getattr(self._report, name)

    def merged_builder(**kwargs):
        report = base(**kwargs)
        return MergedReport(report) if kwargs["loop_result"].hook_blockers else report

    result = run_p3_8_observability_acceptance(
        dependencies(build_report_fn=merged_builder)
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


def assert_gate_failure(receipt, flag, blocker):
    assert (
        receipt.status == "fail"
        and not getattr(receipt, flag)
        and blocker in codes(receipt)
    )


def test_explicit_hook_warning_preserved_by_real_audit():
    assert _audit_hooks(_default_dependencies())


def test_lost_explicit_hook_warning_fails_gate():
    base = _default_dependencies().dispatch_hooks_fn

    def dropped_warning(*args, **kwargs):
        result = base(*args, **kwargs)
        return replace(
            result,
            warnings=tuple(
                issue
                for issue in result.warnings
                if issue.code != "explicit_hook_detail"
            ),
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(
            dependencies(dispatch_hooks_fn=dropped_warning)
        ),
        "hook_contract_valid",
        "p3_8_hook_contract_failed",
    )


def test_disable_action_warning_preserved_by_real_audit():
    assert _audit_hooks(_default_dependencies())


def test_lost_disablement_action_fails_gate():
    base = _default_dependencies().dispatch_hooks_fn

    def dropped_action(*args, **kwargs):
        result = base(*args, **kwargs)
        return replace(
            result,
            warnings=tuple(
                issue
                for issue in result.warnings
                if issue.code != "learning_hook_disabled"
            ),
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(
            dependencies(dispatch_hooks_fn=dropped_action)
        ),
        "hook_contract_valid",
        "p3_8_hook_contract_failed",
    )


def test_checkpoint_order_preserved_by_real_audit():
    assert _audit_run_reporting(_default_dependencies())


def test_reordered_checkpoints_fail_gate():
    base = _default_dependencies().build_report_fn

    def reversed_receipts(**kwargs):
        report = base(**kwargs)
        return replace(
            report,
            checkpoints=replace(
                report.checkpoints,
                receipts=tuple(reversed(report.checkpoints.receipts)),
            ),
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(
            dependencies(build_report_fn=reversed_receipts)
        ),
        "run_reporting_valid",
        "p3_8_run_reporting_failed",
    )


def test_scopes_preserved_by_real_audit():
    assert _audit_run_reporting(_default_dependencies())


def test_corrupted_update_scope_fails_gate():
    base = _default_dependencies().build_report_fn

    def wrong_scope(**kwargs):
        report = base(**kwargs)
        return replace(
            report, scopes=replace(report.scopes, update_scope="named_region")
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(dependencies(build_report_fn=wrong_scope)),
        "run_reporting_valid",
        "p3_8_run_reporting_failed",
    )


def test_corrupted_objective_scope_fails_gate():
    base = _default_dependencies().build_report_fn

    def wrong_scope(**kwargs):
        report = base(**kwargs)
        return replace(
            report, scopes=replace(report.scopes, objective_scope="named_region")
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(dependencies(build_report_fn=wrong_scope)),
        "run_reporting_valid",
        "p3_8_run_reporting_failed",
    )


def test_report_builder_receives_completed_result():
    deps = _default_dependencies()
    observed = []

    def recording_builder(**kwargs):
        result = kwargs["loop_result"]
        observed.append(
            (result.steps_completed, result.global_step, result.stop_reason)
        )
        return deps.build_report_fn(**kwargs)

    result = deps.run_opt_in_loop_with_builder_fn(recording_builder, max_steps=2)
    assert result.report is not None and observed == [(2, 2, "max_steps")]


def test_incomplete_report_input_fails_gate():
    base = _default_dependencies().run_loop_fn

    def incomplete_loop(**kwargs):
        result = base(**kwargs)
        return (
            replace(result, global_step=0)
            if kwargs.get("starting_global_step") == 40
            else result
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(dependencies(run_loop_fn=incomplete_loop)),
        "run_reporting_valid",
        "p3_8_run_reporting_failed",
    )


def test_report_failure_preserves_full_completed_outcome():
    assert _audit_run_reporting(_default_dependencies())


def test_report_failure_outcome_mutation_fails_gate():
    base = _default_dependencies().build_report_fn

    def mutating_failure(**kwargs):
        if kwargs["run_id"] == "":
            kwargs["loop_result"].final_execution.parameters["head.weight"] = 99.0
        return base(**kwargs)

    assert_gate_failure(
        run_p3_8_observability_acceptance(
            dependencies(build_report_fn=mutating_failure)
        ),
        "run_reporting_valid",
        "p3_8_run_reporting_failed",
    )


def test_reported_and_unreported_full_state_equal():
    assert _audit_observer_only_boundary(_default_dependencies())


def test_status_mutation_fails_observer_audit():
    base = _default_dependencies().run_loop_fn

    def changed(**kwargs):
        result = base(**kwargs)
        return (
            replace(result, status="fail") if kwargs.get("emit_run_report") else result
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(dependencies(run_loop_fn=changed)),
        "observer_only_boundary_valid",
        "p3_8_observer_boundary_failed",
    )


def test_global_step_mutation_fails_observer_audit():
    base = _default_dependencies().run_loop_fn

    def changed(**kwargs):
        result = base(**kwargs)
        return (
            replace(result, global_step=result.global_step + 1)
            if kwargs.get("emit_run_report")
            else result
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(dependencies(run_loop_fn=changed)),
        "observer_only_boundary_valid",
        "p3_8_observer_boundary_failed",
    )


def test_batch_count_mutation_fails_observer_audit():
    base = _default_dependencies().run_loop_fn

    def changed(**kwargs):
        result = base(**kwargs)
        return (
            replace(result, batches_consumed=result.batches_consumed + 1)
            if kwargs.get("emit_run_report")
            else result
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(dependencies(run_loop_fn=changed)),
        "observer_only_boundary_valid",
        "p3_8_observer_boundary_failed",
    )


def test_checkpoint_mutation_fails_observer_audit():
    base = _default_dependencies().run_loop_fn

    def changed(**kwargs):
        result = base(**kwargs)
        return (
            replace(result, checkpoints=("wrong",))
            if kwargs.get("emit_run_report")
            else result
        )

    assert_gate_failure(
        run_p3_8_observability_acceptance(dependencies(run_loop_fn=changed)),
        "observer_only_boundary_valid",
        "p3_8_observer_boundary_failed",
    )


def test_report_json_forbidden_state_names_absent():
    assert _audit_observer_only_boundary(_default_dependencies())


def injected_source(fragment):
    base = _default_dependencies().source_loader
    return lambda path: fragment if path.name == "hooks.py" else base(path)


def test_torch_alias_import_fails_full_gate():
    receipt = run_p3_8_observability_acceptance(
        dependencies(source_loader=injected_source("import torch as t"))
    )
    assert_gate_failure(receipt, "import_boundary_valid", "p3_8_import_boundary_failed")


def test_importlib_wandb_fails_full_gate():
    receipt = run_p3_8_observability_acceptance(
        dependencies(
            source_loader=injected_source(
                "import importlib\nimportlib.import_module('wandb')"
            )
        )
    )
    assert_gate_failure(receipt, "import_boundary_valid", "p3_8_import_boundary_failed")


def test_dunder_requests_import_fails_full_gate():
    receipt = run_p3_8_observability_acceptance(
        dependencies(source_loader=injected_source("__import__('requests')"))
    )
    assert_gate_failure(receipt, "import_boundary_valid", "p3_8_import_boundary_failed")


def injected_test_source(fragment):
    base = _default_dependencies().source_loader
    return lambda path: fragment if path.name == "test_hooks.py" else base(path)


def test_assert_true_fails_full_gate():
    receipt = run_p3_8_observability_acceptance(
        dependencies(
            source_loader=injected_test_source(
                "def test_placeholder():\n    assert " + "True\n"
            )
        )
    )
    assert_gate_failure(receipt, "test_inventory_valid", "p3_8_test_inventory_failed")


def test_skip_call_fails_full_gate():
    receipt = run_p3_8_observability_acceptance(
        dependencies(
            source_loader=injected_test_source(
                "def test_placeholder():\n    pytest" + ".skip('x')\n"
            )
        )
    )
    assert_gate_failure(receipt, "test_inventory_valid", "p3_8_test_inventory_failed")


def test_imported_test_reuse_fails_full_gate():
    source = (
        "from "
        + "tests.other import "
        + "test_other\ndef test_placeholder():\n    return 1\n"
    )
    receipt = run_p3_8_observability_acceptance(
        dependencies(source_loader=injected_test_source(source))
    )
    assert_gate_failure(receipt, "test_inventory_valid", "p3_8_test_inventory_failed")


def test_called_test_reuse_fails_full_gate():
    receipt = run_p3_8_observability_acceptance(
        dependencies(
            source_loader=injected_test_source(
                "def test_placeholder():\n    test_other()\n"
            )
        )
    )
    assert_gate_failure(receipt, "test_inventory_valid", "p3_8_test_inventory_failed")


def test_pass_placeholder_fails_full_gate():
    receipt = run_p3_8_observability_acceptance(
        dependencies(
            source_loader=injected_test_source(
                "def test_placeholder():\n    " + "pass\n"
            )
        )
    )
    assert_gate_failure(receipt, "test_inventory_valid", "p3_8_test_inventory_failed")
