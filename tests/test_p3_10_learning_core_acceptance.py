from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from radjax_student import architecture as _architecture
from radjax_student.learning import LearningIssue
from radjax_student.learning.observability_acceptance import (
    _default_dependencies as p38_default_dependencies,
)
from radjax_student.learning.observability_acceptance import (
    run_p3_8_observability_acceptance,
)
from radjax_student.learning.p3_10_acceptance import (
    CLAIMS,
    NON_CLAIMS,
    SCHEMA,
    VALIDITY_FIELDS,
    P310LearningCoreAcceptanceReceipt,
    _default_dependencies,
    _GoldenObjective,
    main,
    run_p3_10_learning_core_acceptance,
)
from radjax_student.learning.synthetic_smoke import (
    P39SmokeDependencies,
    run_p3_9_synthetic_learning_smoke,
)
from radjax_student.optimizers import (
    GradientTree,
    OptimizerConfig,
    SgdOptimizer,
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


def test_02_schema_exact():
    assert receipt().schema_version == SCHEMA


def test_03_all_flags_pass():
    assert all(getattr(receipt(), name) for name in VALIDITY_FIELDS)


def test_04_no_blockers():
    assert receipt().blockers == ()


def test_05_receipt_immutable():
    with pytest.raises((AttributeError, TypeError)):
        receipt().status = "fail"


def test_06_dict_deterministic():
    assert receipt().to_dict() == receipt().to_dict()


def test_07_json_deterministic():
    assert receipt().to_json() == receipt().to_json()


def test_08_claims_exact():
    assert receipt().claims_made == CLAIMS


def test_09_nonclaims_complete():
    assert set(NON_CLAIMS).issubset(receipt().claims_not_made)


def test_10_pass_requires_flags():
    with pytest.raises(ValueError):
        valid_receipt(contracts_valid=False)


def test_11_pass_rejects_blocker():
    with pytest.raises(ValueError):
        valid_receipt(blockers=(LearningIssue("x", "x"),))


def test_12_fail_requires_evidence():
    with pytest.raises(ValueError):
        valid_receipt(status="fail")


def test_13_bad_schema_rejected():
    with pytest.raises(ValueError):
        valid_receipt(schema_version="wrong")


def test_14_nonboolean_rejected():
    with pytest.raises(TypeError):
        valid_receipt(loop_valid=1)


def test_15_bad_issue_rejected():
    with pytest.raises(TypeError):
        valid_receipt(blockers=(object(),))


def test_16_duplicate_claim_rejected():
    with pytest.raises(ValueError):
        valid_receipt(claims_made=CLAIMS + (CLAIMS[0],))


def test_17_metadata_is_gate_identified():
    assert receipt().metadata["gate"] == "P3.10.1"


def test_18_section_count_exact():
    assert receipt().metadata["section_count"] == 11


def test_19_contract_audit_passes():
    assert receipt().contracts_valid


def test_20_architecture_audit_passes():
    assert receipt().contracts_valid


def test_21_optimizer_audit_passes():
    assert receipt().optimizer_valid


def test_22_single_step_audit_passes():
    assert receipt().single_step_valid


def test_23_loop_audit_passes():
    assert receipt().loop_valid


def test_24_checkpoint_audit_passes():
    assert receipt().checkpoint_valid


def test_25_resume_audit_passes():
    assert receipt().resume_valid


def test_26_observability_audit_passes():
    assert receipt().observability_valid


def test_27_synthetic_audit_passes():
    assert receipt().synthetic_learning_valid


def test_28_replay_audit_passes():
    assert receipt().deterministic_replay_valid


def test_29_documentation_audit_passes():
    assert receipt().documentation_valid


def test_30_inventory_audit_passes():
    assert receipt().test_inventory_valid


def test_31_architecture_region_tamper_detected():
    base = _default_dependencies().architecture_factory

    class Leaky:
        def __init__(self):
            self.inner = base()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def resolve_update_scope(self, scope, catalog):
            result = self.inner.resolve_update_scope(scope, catalog)
            if scope.region_id == "trunk":
                return replace(
                    result,
                    selected_parameter_paths=catalog.paths,
                    excluded_parameter_paths=(),
                )
            return result

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), architecture_factory=Leaky)
    )
    assert result.status == "fail" and "p3_10_contracts_failed" in codes(result)


def test_32_architecture_capability_tamper_detected():
    base = _default_dependencies().architecture_factory

    class MissingCapability:
        def __init__(self):
            self.inner = base()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def capability_profile(self):
            profile = self.inner.capability_profile()
            return replace(
                profile,
                capabilities=tuple(
                    x for x in profile.capabilities if x != "architecture.forward_v1"
                ),
            )

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), architecture_factory=MissingCapability)
    )
    assert result.status == "fail" and "p3_10_contracts_failed" in codes(result)


def test_33_architecture_catalog_tamper_detected():
    base = _default_dependencies().architecture_factory

    class MissingPath:
        def __init__(self):
            self.inner = base()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def describe_parameters(self, parameters=None):
            return _architecture.ParameterCatalog(
                self.inner.architecture_id,
                (self.inner.describe_parameters().parameters[0],),
            )

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), architecture_factory=MissingPath)
    )
    assert result.status == "fail" and "p3_10_contracts_failed" in codes(result)


def test_34_architecture_forward_tamper_detected():
    base = _default_dependencies().architecture_factory

    class BadForward:
        def __init__(self):
            self.inner = base()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def forward(self, request):
            return replace(self.inner.forward(request), outputs=(999,))

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), architecture_factory=BadForward)
    )
    assert result.status == "fail" and "p3_10_contracts_failed" in codes(result)


def test_35_architecture_objective_tamper_detected():
    base = _default_dependencies().architecture_factory

    class BadObjective:
        def __init__(self):
            self.inner = base()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def resolve_objective_scope(self, scope, metadata):
            return replace(
                self.inner.resolve_objective_scope(scope, metadata), surface_id="wrong"
            )

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), architecture_factory=BadObjective)
    )
    assert result.status == "fail" and "p3_10_contracts_failed" in codes(result)


def test_36_optimizer_excluded_parameter_tamper_detected():
    base = _default_dependencies().optimizer_factory

    class LeakyOptimizer:
        def __init__(self):
            self.inner = base()
            self.optimizer_id = self.inner.optimizer_id

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def apply_updates(self, request):
            result = self.inner.apply_updates(request)
            if request.resolved_update_selection.selected_parameter_paths == (
                "trunk.weight",
            ):
                return replace(
                    result,
                    updated_parameters={**result.updated_parameters, "head.bias": 1.0},
                )
            return result

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), optimizer_factory=LeakyOptimizer)
    )
    assert result.status == "fail" and "p3_10_optimizer_failed" in codes(result)


class OptimizerWrapperForTest:
    def __init__(self):
        self.inner = SgdOptimizer()

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ExcludedStateOptimizerForTest(OptimizerWrapperForTest):
    def apply_updates(self, request):
        result = self.inner.apply_updates(request)
        backend_state = dict(result.updated_optimizer_state.backend_state)
        per_parameter_steps = dict(backend_state["per_parameter_steps"])
        per_parameter_steps["head.bias"] += 1
        backend_state["per_parameter_steps"] = per_parameter_steps
        return replace(
            result,
            updated_optimizer_state=replace(
                result.updated_optimizer_state, backend_state=backend_state
            ),
        )


class DoubleStepOptimizerForTest(OptimizerWrapperForTest):
    def apply_updates(self, request):
        result = self.inner.apply_updates(request)
        return replace(
            result,
            updated_optimizer_state=replace(
                result.updated_optimizer_state,
                step=result.updated_optimizer_state.step + 2,
            ),
        )


class IgnoredLearningRateOptimizerForTest(OptimizerWrapperForTest):
    def apply_updates(self, request):
        return self.inner.apply_updates(replace(request, schedule_values={}))


class MismatchAcceptingOptimizerForTest(OptimizerWrapperForTest):
    def apply_updates(self, request):
        if request.config.optimizer_id == "wrong.optimizer":
            values = vars(request).copy()
            values["config"] = OptimizerConfig(
                self.inner.optimizer_id, learning_rate=0.25
            )
            request = SimpleNamespace(**values)
        return self.inner.apply_updates(request)


class NonFiniteAcceptingOptimizerForTest(OptimizerWrapperForTest):
    def apply_updates(self, request):
        values = dict(request.gradients.values)
        if any(not math.isfinite(float(value)) for value in values.values()):
            request = replace(
                request,
                gradients=GradientTree(
                    request.gradients.parameter_paths,
                    values={
                        path: 0.0 if not math.isfinite(float(value)) else value
                        for path, value in values.items()
                    },
                ),
            )
        return self.inner.apply_updates(request)


def test_37_optimizer_excluded_state_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(), optimizer_factory=ExcludedStateOptimizerForTest
        )
    )
    assert result.status == "fail" and "p3_10_optimizer_failed" in codes(result)


def test_38_optimizer_step_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), optimizer_factory=DoubleStepOptimizerForTest)
    )
    assert result.status == "fail" and "p3_10_optimizer_failed" in codes(result)


def test_39_optimizer_learning_rate_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            optimizer_factory=IgnoredLearningRateOptimizerForTest,
        )
    )
    assert result.status == "fail" and "p3_10_optimizer_failed" in codes(result)


def test_40_optimizer_mismatch_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(), optimizer_factory=MismatchAcceptingOptimizerForTest
        )
    )
    assert result.status == "fail" and "p3_10_optimizer_failed" in codes(result)


def test_41_optimizer_nonfinite_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            optimizer_factory=NonFiniteAcceptingOptimizerForTest,
        )
    )
    assert result.status == "fail" and "p3_10_optimizer_failed" in codes(result)


def test_42_optimizer_replay_is_deterministic():
    assert receipt().deterministic_replay_valid


def test_43_single_step_tamper_detected():
    base = _default_dependencies().single_step_fn

    def bad_step(**kwargs):
        result = base(**kwargs)
        return replace(
            result, learning_state=replace(result.learning_state, global_step=2)
        )

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), single_step_fn=bad_step)
    )
    assert result.status == "fail" and "p3_10_single_step_failed" in codes(result)


class AcceptingStepFailuresForTest:
    def __init__(self):
        self.inner = _default_dependencies().single_step_fn

    def __call__(self, **kwargs):
        try:
            return self.inner(**kwargs)
        except Exception:
            return self.inner(**{**kwargs, "objective": _GoldenObjective()})


class LeakySingleStepParameterForTest:
    def __init__(self):
        self.inner = _default_dependencies().single_step_fn

    def __call__(self, **kwargs):
        result = self.inner(**kwargs)
        return replace(
            result,
            parameters={**result.parameters, "head.bias": 1.0},
        )


class LeakySingleStepStateForTest:
    def __init__(self):
        self.inner = _default_dependencies().single_step_fn

    def __call__(self, **kwargs):
        result = self.inner(**kwargs)
        backend_state = dict(result.optimizer_state.backend_state)
        steps = dict(backend_state["per_parameter_steps"])
        steps["head.bias"] += 1
        backend_state["per_parameter_steps"] = steps
        return replace(
            result,
            optimizer_state=replace(
                result.optimizer_state, backend_state=backend_state
            ),
        )


class MutatingFailureExecutionForTest:
    def __init__(self):
        self.inner = _default_dependencies().single_step_fn

    def __call__(self, **kwargs):
        result = self.inner(**kwargs)
        return replace(
            result,
            result=replace(result.result, status="fail"),
            parameters={**result.parameters, "head.bias": 1.0},
        )


def test_44_single_step_invalid_gradient_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            single_step_fn=AcceptingStepFailuresForTest(),
        )
    )
    assert result.status == "fail" and "p3_10_single_step_failed" in codes(result)


def test_45_single_step_excluded_parameter_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            single_step_fn=LeakySingleStepParameterForTest(),
        )
    )
    assert result.status == "fail" and "p3_10_single_step_failed" in codes(result)


def test_46_single_step_excluded_state_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), single_step_fn=LeakySingleStepStateForTest())
    )
    assert result.status == "fail" and "p3_10_single_step_failed" in codes(result)


def test_47_single_step_nonfinite_loss_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            single_step_fn=AcceptingStepFailuresForTest(),
        )
    )
    assert result.status == "fail" and "p3_10_single_step_failed" in codes(result)


def test_48_single_step_mutating_failure_execution_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            single_step_fn=MutatingFailureExecutionForTest(),
        )
    )
    assert result.status == "fail" and "p3_10_single_step_failed" in codes(result)


def test_49_loop_wrong_consumption_tamper_detected():
    base = _default_dependencies().run_loop_fn

    def bad_loop(**kwargs):
        return replace(base(**kwargs), batches_consumed=3)

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), run_loop_fn=bad_loop)
    )
    assert result.status == "fail" and "p3_10_loop_failed" in codes(result)


def test_50_loop_global_step_tamper_detected():
    base = _default_dependencies().run_loop_fn

    def bad_loop(**kwargs):
        return replace(base(**kwargs), global_step=99)

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), run_loop_fn=bad_loop)
    )
    assert result.status == "fail" and "p3_10_loop_failed" in codes(result)


class ReorderedCheckpointLoopForTest:
    def __init__(self):
        self.inner = _default_dependencies().run_loop_fn

    def __call__(self, **kwargs):
        result = self.inner(**kwargs)
        return replace(result, checkpoints=tuple(reversed(result.checkpoints)))


class ContinuingAfterFailureLoopForTest:
    def __init__(self):
        self.inner = _default_dependencies().run_loop_fn

    def __call__(self, **kwargs):
        result = self.inner(**kwargs)
        if result.stop_reason == "learning_step_failure":
            return replace(
                result,
                steps_completed=2,
                global_step=2,
                batches_consumed=2,
            )
        return result


class MislabelledExhaustionLoopForTest:
    def __init__(self):
        self.inner = _default_dependencies().run_loop_fn

    def __call__(self, **kwargs):
        result = self.inner(**kwargs)
        if kwargs["batch_source"].source_id == "p310.exhausted":
            return replace(result, status="pass", stop_reason="max_steps")
        return result


def test_51_loop_checkpoint_cadence_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), run_loop_fn=ReorderedCheckpointLoopForTest())
    )
    assert result.status == "fail" and "p3_10_loop_failed" in codes(result)


def test_52_loop_continue_after_failure_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(), run_loop_fn=ContinuingAfterFailureLoopForTest()
        )
    )
    assert result.status == "fail" and "p3_10_loop_failed" in codes(result)


def test_53_loop_exhaustion_label_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            run_loop_fn=MislabelledExhaustionLoopForTest(),
        )
    )
    assert result.status == "fail" and "p3_10_loop_failed" in codes(result)


def test_54_loop_resumed_start_surface_audited():
    assert receipt().loop_valid


class CheckpointLoadTamperForTest:
    def __init__(self, target=None, mode=None):
        self.base = _default_dependencies().checkpoint_load_fn
        self.target = target
        self.mode = mode
        self.good_path = None

    def __call__(self, directory, *, runtime_reference=None):
        directory = Path(directory)
        if self.good_path is None and runtime_reference == "p310-runtime":
            self.good_path = directory
            return self.base(directory, runtime_reference=runtime_reference)
        if self.mode == "runtime" and directory == self.good_path:
            return self.base(self.good_path, runtime_reference="p310-runtime")
        if directory.name == self.target:
            loaded = self.base(self.good_path, runtime_reference="p310-runtime")
            if self.mode == "none":
                return replace(loaded, source_state={"position": 1})
            return loaded
        return self.base(directory, runtime_reference=runtime_reference)


def test_55_checkpoint_source_hash_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_load_fn=CheckpointLoadTamperForTest("tamper-3"),
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)


def test_56_checkpoint_source_size_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_load_fn=CheckpointLoadTamperForTest("tamper-4"),
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)


def test_57_checkpoint_runtime_mismatch_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_load_fn=CheckpointLoadTamperForTest(mode="runtime"),
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)


def test_58_checkpoint_manifest_ownership_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_load_fn=CheckpointLoadTamperForTest("tamper-5"),
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)


def test_59_checkpoint_none_source_state_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_load_fn=CheckpointLoadTamperForTest(
                "none-source-state", mode="none"
            ),
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)


def _broken_observability():
    class BrokenSeries:
        def __init__(self, *args, **kwargs):
            self.records = ()

        def add(self, record):
            return None

        def summary(self):
            return type("Summary", (), {"count": 0, "last": 0, "mean": 0, "total": 0})()

    deps = replace(p38_default_dependencies(), metric_series_factory=BrokenSeries)
    return run_p3_8_observability_acceptance(deps)


def test_60_real_observability_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(), observability_acceptance_fn=_broken_observability
        )
    )
    assert result.status == "fail" and "p3_10_observability_failed" in codes(result)


def _broken_source_smoke():
    base = P39SmokeDependencies()

    def altered(path, **kwargs):
        checkpoint = base.checkpoint_restore_fn(path, **kwargs)
        return replace(
            checkpoint, source_state={**checkpoint.source_state, "position": 0}
        )

    return run_p3_9_synthetic_learning_smoke(
        replace(base, checkpoint_restore_fn=altered)
    )


def test_61_real_synthetic_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), synthetic_smoke_fn=_broken_source_smoke)
    )
    assert result.status == "fail" and "p3_10_synthetic_learning_failed" in codes(
        result
    )


def test_62_resume_uses_real_smoke():
    assert receipt().resume_valid


def test_63_synthetic_uses_real_smoke():
    assert receipt().synthetic_learning_valid


def test_64_downstream_receipts_have_no_blockers():
    assert receipt().resume_valid and receipt().synthetic_learning_valid


def test_65_documentation_missing_file_tamper_detected():
    base = _default_dependencies().path_exists_fn
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            path_exists_fn=lambda path: (
                False if path.name == "README.md" else base(path)
            ),
        )
    )
    assert result.status == "fail" and "p3_10_documentation_failed" in codes(result)


def test_66_documentation_marker_tamper_detected():
    base = _default_dependencies().source_loader
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            source_loader=lambda path: (
                base(path).replace("P3.10", "")
                if path.name == "README.md"
                else base(path)
            ),
        )
    )
    assert result.status == "fail" and "p3_10_documentation_failed" in codes(result)


def test_67_inventory_is_ast_based():
    assert receipt().test_inventory_valid


def test_68_inventory_requires_p310_file():
    base = _default_dependencies().path_exists_fn
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            path_exists_fn=lambda path: (
                False
                if path.name == "test_p3_10_learning_core_acceptance.py"
                else base(path)
            ),
        )
    )
    assert result.status == "fail" and "p3_10_test_inventory_failed" in codes(result)


def test_69_inventory_detects_placeholder_fixture():
    base = _default_dependencies().source_loader
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            source_loader=lambda path: (
                "def test_fixture():\n    pass\n"
                if path.name == "test_p3_10_learning_core_acceptance.py"
                else base(path)
            ),
        )
    )
    assert result.status == "fail" and "p3_10_test_inventory_failed" in codes(result)


def test_70_replay_is_exact():
    assert receipt().deterministic_replay_valid


def test_71_replay_tamper_fails_closed():
    calls = 0
    base = _default_dependencies().synthetic_smoke_fn

    def alternate():
        nonlocal calls
        calls += 1
        return _broken_source_smoke() if calls == 4 else base()

    result = run_p3_10_learning_core_acceptance(
        replace(_default_dependencies(), synthetic_smoke_fn=alternate)
    )
    assert result.status == "fail" and "p3_10_deterministic_replay_failed" in codes(
        result
    )


def test_72_cli_passes():
    assert main([]) == 0


def test_73_cli_json_passes():
    assert main(["--json"]) == 0


def test_74_cli_injected_failure_is_nonzero():
    dependencies = replace(
        _default_dependencies(),
        architecture_factory=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    assert main([], dependencies) != 0


def test_75_public_lazy_export_resolves():
    from radjax_student.learning import (
        P310LearningCoreAcceptanceReceipt,
        run_p3_10_learning_core_acceptance,
    )

    assert (
        P310LearningCoreAcceptanceReceipt is not None
        and run_p3_10_learning_core_acceptance().status == "pass"
    )


def test_76_receipt_has_no_tracebacks():
    assert "traceback" not in receipt().to_json().lower()


def test_77_checkpoint_schema_tamper_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_load_fn=CheckpointLoadTamperForTest("tamper-6"),
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)


def test_78_checkpoint_altered_source_state_detected():
    result = run_p3_10_learning_core_acceptance(
        replace(
            _default_dependencies(),
            checkpoint_load_fn=CheckpointLoadTamperForTest(
                "tamper-altered-source-state"
            ),
        )
    )
    assert result.status == "fail" and "p3_10_checkpoint_failed" in codes(result)
