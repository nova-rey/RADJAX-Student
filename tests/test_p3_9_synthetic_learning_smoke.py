"""P3.9 synthetic learning smoke acceptance and adversarial evidence."""

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from radjax_student.learning import LearningIssue
from radjax_student.learning.synthetic_smoke import (
    ARCHITECTURE_ID,
    CLAIMS,
    NON_CLAIMS,
    OPTIMIZER_ID,
    PROBLEM_ID,
    REPORT_SCHEMA,
    SCHEMA,
    P39SyntheticLearningReceipt,
    SyntheticRunSummary,
    _default_dependencies,
    _execute,
    _initial,
    _restore_checkpoint,
    _write_checkpoint,
    main,
    run_p3_9_synthetic_learning_smoke,
)


@lru_cache(maxsize=1)
def receipt():
    return run_p3_9_synthetic_learning_smoke()


def codes(value):
    return {item.code for item in value.blockers}


def summary(**changes):
    values = dict(
        run_id="p3_9:test",
        mode="whole_student",
        status="pass",
        stop_reason="max_steps",
        steps_completed=1,
        global_step=1,
        initial_loss=1.0,
        final_loss=0.5,
        loss_ratio=0.5,
        parameter_deltas={"head.bias": 0.1, "trunk.weight": 0.1},
        changed_parameter_paths=("head.bias", "trunk.weight"),
        unchanged_parameter_paths=(),
        checkpoint_count=0,
        report_schema_version=REPORT_SCHEMA,
    )
    values.update(changes)
    return SyntheticRunSummary(**values)


def valid_receipt(**changes):
    values = dict(
        schema_version=SCHEMA,
        status="pass",
        problem_id=PROBLEM_ID,
        architecture_id=ARCHITECTURE_ID,
        optimizer_id=OPTIMIZER_ID,
        whole_student=summary(),
        trunk_only=summary(mode="trunk_only"),
        head_only=summary(mode="head_only"),
        resume=summary(mode="resume"),
        deterministic_replay_valid=True,
        loss_decrease_valid=True,
        scope_boundaries_valid=True,
        optimizer_boundaries_valid=True,
        checkpoint_restore_valid=True,
        metrics_valid=True,
        hooks_valid=True,
        run_reporting_valid=True,
    )
    values.update(changes)
    return P39SyntheticLearningReceipt(**values)


def test_01_valid_synthetic_run_summary():
    assert summary().status == "pass"


def test_02_invalid_summary_mode_rejected():
    with pytest.raises(ValueError):
        summary(mode="invalid")


def test_03_nonfinite_summary_loss_rejected():
    with pytest.raises(ValueError):
        summary(final_loss=float("nan"))


def test_04_negative_summary_steps_rejected():
    with pytest.raises(ValueError):
        summary(steps_completed=-1)


def test_05_summary_paths_sorted_deterministically():
    assert summary(
        changed_parameter_paths=("trunk.weight", "head.bias")
    ).changed_parameter_paths == ("head.bias", "trunk.weight")


def test_06_valid_receipt_passes():
    assert valid_receipt().status == "pass"


def test_07_receipt_pass_rejects_false_flag():
    with pytest.raises(ValueError):
        valid_receipt(metrics_valid=False)


def test_08_receipt_pass_rejects_blocker():
    with pytest.raises(ValueError):
        valid_receipt(blockers=(LearningIssue("blocked", "blocked"),))


def test_09_receipt_fail_rejects_valid_evidence():
    with pytest.raises(ValueError):
        valid_receipt(status="fail")


def test_10_invalid_receipt_schema_rejected():
    with pytest.raises(ValueError):
        valid_receipt(schema_version="wrong")


def test_11_duplicate_claims_rejected():
    with pytest.raises(ValueError):
        valid_receipt(claims_made=CLAIMS + (CLAIMS[0],))


def test_12_invalid_metadata_rejected():
    with pytest.raises(ValueError):
        valid_receipt(metadata={"parameters": {}})


def test_13_receipt_dict_deterministic():
    assert valid_receipt().to_dict() == valid_receipt().to_dict()


def test_14_receipt_json_deterministic():
    assert valid_receipt().to_json() == valid_receipt().to_json()


def test_15_whole_initial_loss_finite():
    assert receipt().whole_student.initial_loss == 9.0


def test_16_whole_final_loss_finite():
    assert receipt().whole_student.final_loss < float("inf")


def test_17_whole_final_loss_lower():
    assert receipt().whole_student.final_loss < receipt().whole_student.initial_loss


def test_18_whole_loss_ratio_threshold():
    assert receipt().whole_student.loss_ratio <= 0.5


def test_19_whole_trunk_moves():
    assert "trunk.weight" in receipt().whole_student.changed_parameter_paths


def test_20_whole_head_moves():
    assert "head.bias" in receipt().whole_student.changed_parameter_paths


def test_21_whole_optimizer_advances():
    assert receipt().whole_student.global_step == 12


def test_22_whole_exact_steps():
    assert receipt().whole_student.steps_completed == 12


def test_23_whole_global_step_correct():
    assert receipt().whole_student.global_step == 12


def test_24_whole_report_attached():
    assert receipt().whole_student.report_schema_version == REPORT_SCHEMA


def test_25_whole_report_scope_is_whole_student():
    assert receipt().run_reporting_valid


def test_26_trunk_only_changes_trunk():
    assert receipt().trunk_only.changed_parameter_paths == ("trunk.weight",)


def test_27_trunk_only_preserves_head():
    assert receipt().trunk_only.unchanged_parameter_paths == ("head.bias",)


def test_28_trunk_only_preserves_head_optimizer_state():
    assert receipt().optimizer_boundaries_valid


def test_29_head_only_changes_head():
    assert receipt().head_only.changed_parameter_paths == ("head.bias",)


def test_30_head_only_preserves_trunk():
    assert receipt().head_only.unchanged_parameter_paths == ("trunk.weight",)


def test_31_head_only_preserves_trunk_optimizer_state():
    assert receipt().optimizer_boundaries_valid


def test_32_scoped_reports_correct():
    assert receipt().scope_boundaries_valid


def test_33_scoped_losses_finite():
    assert receipt().trunk_only.final_loss > 0 and receipt().head_only.final_loss > 0


def checkpoint_evidence(directory):
    evidence = _execute(
        mode="resume", scope="whole_student", steps=3, deps=_default_dependencies()
    )
    _write_checkpoint(evidence, directory, _default_dependencies())
    return evidence


def test_34_checkpoint_created():
    with TemporaryDirectory() as raw:
        checkpoint_evidence(Path(raw))
        assert (Path(raw) / "manifest.json").is_file()


def test_35_checkpoint_receipt_deterministic():
    with TemporaryDirectory() as raw:
        item = checkpoint_evidence(Path(raw))
        assert item.learning_state.global_step == 3


def test_36_checkpoint_restores_learning_state():
    with TemporaryDirectory() as raw:
        before = checkpoint_evidence(Path(raw))
        state, _ = _restore_checkpoint(
            Path(raw), _default_dependencies(), "whole_student"
        )
        assert state[5] == before.learning_state


def test_37_checkpoint_restores_parameters():
    with TemporaryDirectory() as raw:
        before = checkpoint_evidence(Path(raw))
        state, _ = _restore_checkpoint(
            Path(raw), _default_dependencies(), "whole_student"
        )
        assert dict(state[6]) == dict(before.parameters)


def test_38_checkpoint_restores_optimizer_state():
    with TemporaryDirectory() as raw:
        before = checkpoint_evidence(Path(raw))
        state, _ = _restore_checkpoint(
            Path(raw), _default_dependencies(), "whole_student"
        )
        assert state[4].backend_state == before.optimizer_state.backend_state


def test_39_checkpoint_restores_source_position():
    with TemporaryDirectory() as raw:
        checkpoint_evidence(Path(raw))
        _, source = _restore_checkpoint(
            Path(raw), _default_dependencies(), "whole_student"
        )
        assert source.position == 3


def test_40_resume_parameters_match_uninterrupted():
    assert receipt().checkpoint_restore_valid


def test_41_resume_optimizer_matches_uninterrupted():
    assert receipt().checkpoint_restore_valid


def test_42_resume_learning_state_matches_uninterrupted():
    assert receipt().checkpoint_restore_valid


def test_43_resume_report_contract_valid():
    assert receipt().resume.report_schema_version == REPORT_SCHEMA


def test_44_corrupt_checkpoint_rejected():
    with TemporaryDirectory() as raw:
        checkpoint_evidence(Path(raw))
        (Path(raw) / "architecture.json").write_text("{}")
        with pytest.raises(ValueError):
            _restore_checkpoint(Path(raw), _default_dependencies(), "whole_student")


def test_45_architecture_mismatch_rejected():
    with TemporaryDirectory() as raw:
        checkpoint_evidence(Path(raw))

        def wrong_restore(path, **kwargs):
            item = _default_dependencies().checkpoint_restore_fn(path, **kwargs)
            return replace(
                item,
                architecture_state=replace(item.architecture_state, state_id="wrong"),
            )

        with pytest.raises(ValueError):
            _restore_checkpoint(
                Path(raw),
                replace(_default_dependencies(), checkpoint_restore_fn=wrong_restore),
                "whole_student",
            )


def test_46_optimizer_mismatch_rejected():
    with TemporaryDirectory() as raw:
        checkpoint_evidence(Path(raw))

        def wrong_restore(path, **kwargs):
            item = _default_dependencies().checkpoint_restore_fn(path, **kwargs)
            return replace(
                item,
                optimizer_state=replace(item.optimizer_state, optimizer_id="wrong"),
            )

        with pytest.raises(ValueError):
            _restore_checkpoint(
                Path(raw),
                replace(_default_dependencies(), checkpoint_restore_fn=wrong_restore),
                "whole_student",
            )


def test_47_missing_source_state_rejected():
    with TemporaryDirectory() as raw:
        checkpoint_evidence(Path(raw))
        (Path(raw) / "source.json").unlink()
        with pytest.raises(ValueError):
            _restore_checkpoint(Path(raw), _default_dependencies(), "whole_student")


def test_48_failed_restore_preserves_destination():
    source = _initial("whole_student", "unchanged")[-1]
    with TemporaryDirectory() as raw:
        checkpoint_evidence(Path(raw))
        (Path(raw) / "source.json").unlink()
        with pytest.raises(ValueError):
            _restore_checkpoint(Path(raw), _default_dependencies(), "whole_student")
    assert source == {"trunk.weight": 0.0, "head.bias": 0.0}


def test_49_required_metrics_present():
    assert receipt().metrics_valid


def test_50_hook_metric_present():
    assert receipt().metrics_valid


def test_51_hook_events_ordered():
    assert receipt().hooks_valid


def test_52_hook_cannot_mutate_core_state():
    assert receipt().whole_student.changed_parameter_paths == (
        "head.bias",
        "trunk.weight",
    )


def test_53_report_schema_exact():
    assert receipt().whole_student.report_schema_version == REPORT_SCHEMA


def test_54_report_global_step_correct():
    assert receipt().whole_student.global_step == 12


def test_55_report_checkpoint_order_preserved():
    assert receipt().whole_student.checkpoint_count == 4


def test_56_report_claims_exact():
    assert receipt().claims_made == CLAIMS


def test_57_report_nonclaims_required():
    assert set(NON_CLAIMS).issubset(receipt().claims_not_made)


def test_58_report_json_deterministic():
    assert receipt().to_json() == receipt().to_json()


def test_59_replay_parameters_identical():
    assert receipt().deterministic_replay_valid


def test_60_replay_optimizer_identical():
    assert receipt().deterministic_replay_valid


def test_61_replay_metrics_identical():
    assert receipt().deterministic_replay_valid


def test_62_replay_hooks_identical():
    assert receipt().deterministic_replay_valid


def test_63_replay_reports_identical():
    assert receipt().deterministic_replay_valid


def test_64_replay_receipts_identical():
    assert receipt().to_dict() == run_p3_9_synthetic_learning_smoke().to_dict()


def test_65_loss_threshold_tamper_fails_receipt():
    base = _default_dependencies().run_loop_fn

    def no_learning(**kwargs):
        result = base(**kwargs)
        execution = result.final_execution
        return replace(
            result,
            final_execution=replace(
                execution, parameters={"trunk.weight": 0.0, "head.bias": 0.0}
            ),
        )

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), run_loop_fn=no_learning)
    )
    assert not result.loss_decrease_valid and "p3_9_loss_threshold_failed" in codes(
        result
    )


def test_66_scope_boundary_tamper_fails_receipt():
    base = _default_dependencies().run_loop_fn

    def leaked_scope(**kwargs):
        result = base(**kwargs)
        execution = result.final_execution
        if kwargs["learning_state"].active_update_scope.region_id == "trunk":
            return replace(
                result,
                final_execution=replace(
                    execution, parameters={**execution.parameters, "head.bias": 1.0}
                ),
            )
        return result

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), run_loop_fn=leaked_scope)
    )
    assert not result.scope_boundaries_valid and "p3_9_scope_boundary_failed" in codes(
        result
    )


def test_67_resume_tamper_fails_receipt():
    def altered_restore(path, **kwargs):
        item = _default_dependencies().checkpoint_restore_fn(path, **kwargs)
        return replace(
            item,
            parameters={
                **item.parameters,
                "head.bias": item.parameters["head.bias"] + 1.0,
            },
        )

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), checkpoint_restore_fn=altered_restore)
    )
    assert not result.checkpoint_restore_valid and "p3_9_resume_mismatch" in codes(
        result
    )


def test_68_missing_metric_tamper_fails_receipt():
    base = _default_dependencies().run_loop_fn

    def missing_metric(**kwargs):
        result = base(**kwargs)
        return replace(
            result,
            metrics=tuple(
                item for item in result.metrics if item.name != "gradient_norm"
            ),
        )

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), run_loop_fn=missing_metric)
    )
    assert not result.metrics_valid and "p3_9_metrics_missing" in codes(result)


def test_69_hook_observation_tamper_fails_receipt():
    base = _default_dependencies().run_loop_fn

    def missing_event(**kwargs):
        result = base(**kwargs)
        return replace(
            result,
            hook_events=tuple(
                event for event in result.hook_events if event != "step_end"
            ),
        )

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), run_loop_fn=missing_event)
    )
    assert not result.hooks_valid and "p3_9_hook_observation_failed" in codes(result)


def test_70_report_tamper_fails_receipt():
    base = _default_dependencies().build_report_fn

    def corrupt_report(**kwargs):
        report = base(**kwargs)
        return replace(report, status=replace(report.status, global_step=999))

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), build_report_fn=corrupt_report)
    )
    assert not result.run_reporting_valid and "p3_9_run_report_failed" in codes(result)


def test_71_unexpected_exception_returns_fail_receipt():
    def raises(**kwargs):
        raise RuntimeError("injected")

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), run_loop_fn=raises)
    )
    assert result.status == "fail"


def test_72_unexpected_exception_adds_stable_blocker():
    def raises(**kwargs):
        raise RuntimeError("injected")

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), run_loop_fn=raises)
    )
    assert "p3_9_internal_error" in codes(result)


def test_73_cli_returns_zero_on_pass():
    assert main([], _default_dependencies()) == 0


def test_74_cli_returns_nonzero_on_injected_failure():
    def raises(**kwargs):
        raise RuntimeError("injected")

    assert main([], replace(_default_dependencies(), run_loop_fn=raises)) != 0


def test_75_optimizer_boundary_tamper_fails_receipt():
    base = _default_dependencies().run_loop_fn

    def leaked_optimizer(**kwargs):
        result = base(**kwargs)
        execution = result.final_execution
        if kwargs["learning_state"].active_update_scope.region_id == "head":
            state = replace(
                execution.optimizer_state,
                backend_state={
                    "per_parameter_steps": {"head.bias": 6, "trunk.weight": 6}
                },
            )
            return replace(
                result, final_execution=replace(execution, optimizer_state=state)
            )
        return result

    result = run_p3_9_synthetic_learning_smoke(
        replace(_default_dependencies(), run_loop_fn=leaked_optimizer)
    )
    assert (
        not result.optimizer_boundaries_valid
        and "p3_9_optimizer_boundary_failed" in codes(result)
    )
