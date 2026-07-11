"""Deterministic P3.9 end-to-end proof for the generic learning stack."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from radjax_student import architecture as _architecture
from radjax_student.checkpoints import (
    LearningCheckpoint,
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from radjax_student.learning.hooks import HookContext, HookResult
from radjax_student.learning.models import LearningBatch, LearningState, MetricRecord
from radjax_student.learning.run_report import SCHEMA as REPORT_SCHEMA
from radjax_student.learning.run_report import (
    LearningRunReport,
    build_learning_run_report,
)
from radjax_student.learning.scopes import (
    ResolvedUpdateSelection,
    UpdateScope,
)
from radjax_student.optimizers import (
    OptimizerConfig,
    OptimizerInitRequest,
    SgdOptimizer,
)
from radjax_student.steps.loop import (
    LearningLoopConfig,
    LearningLoopResult,
    SyntheticBatchSource,
    run_learning_loop,
)

ArchitectureCapabilityProfile = _architecture.ArchitectureCapabilityProfile
ArchitectureConfig = _architecture.ArchitectureConfig
ArchitectureInitRequest = _architecture.ArchitectureInitRequest
ArchitectureInitResult = _architecture.ArchitectureInitResult
ArchitectureMetadata = _architecture.ArchitectureMetadata
ArchitectureState = _architecture.ArchitectureState
BatchValidationResult = _architecture.BatchValidationResult
ForwardRequest = _architecture.ForwardRequest
ForwardResult = _architecture.ForwardResult
IntermediateSurfaceDescriptor = _architecture.IntermediateSurfaceDescriptor
NamedRegion = _architecture.NamedRegion
ParameterCatalog = _architecture.ParameterCatalog
ParameterDescriptor = _architecture.ParameterDescriptor
ResolvedObjectiveSelection = _architecture.ResolvedObjectiveSelection

SCHEMA = "radjax.p3_9_synthetic_learning_smoke.v1"
PROBLEM_ID = "synthetic_linear_y_equals_2x_plus_1.v1"
ARCHITECTURE_ID = "synthetic_linear_v1"
OPTIMIZER_ID = "sgd.v1"
SYNTHETIC_NUMERIC_TOLERANCE = 0.0
CLAIMS = (
    "synthetic_learning_end_to_end_validated",
    "whole_student_learning_validated",
    "scoped_update_boundaries_validated",
    "checkpoint_resume_validated",
    "deterministic_replay_validated",
    "p3_8_observability_stack_exercised",
)
NON_CLAIMS = (
    "model_quality",
    "real_architecture_support",
    "tome_training",
    "language_modeling",
    "distributed_training",
    "accelerator_performance",
    "production_hyperparameters",
    "evaluation",
    "generalization",
)
_VALIDITY_FIELDS = (
    "deterministic_replay_valid",
    "loss_decrease_valid",
    "scope_boundaries_valid",
    "optimizer_boundaries_valid",
    "checkpoint_restore_valid",
    "metrics_valid",
    "hooks_valid",
    "run_reporting_valid",
)


@dataclass(frozen=True)
class SyntheticRunSummary:
    run_id: str
    mode: Literal["whole_student", "trunk_only", "head_only", "resume", "replay"]
    status: Literal["pass", "fail"]
    stop_reason: str
    steps_completed: int
    global_step: int
    initial_loss: float
    final_loss: float
    loss_ratio: float
    parameter_deltas: Mapping[str, float]
    changed_parameter_paths: tuple[str, ...]
    unchanged_parameter_paths: tuple[str, ...]
    checkpoint_count: int
    report_schema_version: str | None

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or self.mode
            not in {"whole_student", "trunk_only", "head_only", "resume", "replay"}
            or self.status not in {"pass", "fail"}
            or not self.stop_reason
        ):
            raise ValueError("synthetic run summary identity is invalid")
        if min(self.steps_completed, self.global_step, self.checkpoint_count) < 0:
            raise ValueError("synthetic run counters must be nonnegative")
        if not all(
            math.isfinite(value)
            for value in (self.initial_loss, self.final_loss, self.loss_ratio)
        ):
            raise ValueError("synthetic run losses must be finite")
        deltas = dict(
            sorted(
                (str(path), float(value))
                for path, value in self.parameter_deltas.items()
            )
        )
        if not all(math.isfinite(value) for value in deltas.values()):
            raise ValueError("parameter deltas must be finite")
        changed = tuple(sorted(self.changed_parameter_paths))
        unchanged = tuple(sorted(self.unchanged_parameter_paths))
        if set(changed) & set(unchanged) or set(changed) | set(unchanged) != set(
            deltas
        ):
            raise ValueError("parameter path summaries are inconsistent")
        if self.report_schema_version not in (None, REPORT_SCHEMA):
            raise ValueError("synthetic run report schema is invalid")
        object.__setattr__(self, "parameter_deltas", MappingProxyType(deltas))
        object.__setattr__(self, "changed_parameter_paths", changed)
        object.__setattr__(self, "unchanged_parameter_paths", unchanged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps_completed": self.steps_completed,
            "global_step": self.global_step,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_ratio": self.loss_ratio,
            "parameter_deltas": dict(self.parameter_deltas),
            "changed_parameter_paths": list(self.changed_parameter_paths),
            "unchanged_parameter_paths": list(self.unchanged_parameter_paths),
            "checkpoint_count": self.checkpoint_count,
            "report_schema_version": self.report_schema_version,
        }


@dataclass(frozen=True)
class P39SyntheticLearningReceipt:
    schema_version: str
    status: Literal["pass", "fail"]
    problem_id: str
    architecture_id: str
    optimizer_id: str
    whole_student: SyntheticRunSummary
    trunk_only: SyntheticRunSummary
    head_only: SyntheticRunSummary
    resume: SyntheticRunSummary
    deterministic_replay_valid: bool
    loss_decrease_valid: bool
    scope_boundaries_valid: bool
    optimizer_boundaries_valid: bool
    checkpoint_restore_valid: bool
    metrics_valid: bool
    hooks_valid: bool
    run_reporting_valid: bool
    blockers: tuple[object, ...] = ()
    warnings: tuple[object, ...] = ()
    claims_made: tuple[str, ...] = CLAIMS
    claims_not_made: tuple[str, ...] = NON_CLAIMS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from radjax_student.learning import LearningIssue

        if self.schema_version != SCHEMA or self.status not in {"pass", "fail"}:
            raise ValueError("synthetic receipt schema or status is invalid")
        if (self.problem_id, self.architecture_id, self.optimizer_id) != (
            PROBLEM_ID,
            ARCHITECTURE_ID,
            OPTIMIZER_ID,
        ):
            raise ValueError("synthetic receipt identity is invalid")
        runs = (self.whole_student, self.trunk_only, self.head_only, self.resume)
        if any(not isinstance(run, SyntheticRunSummary) for run in runs):
            raise TypeError("synthetic receipt runs are invalid")
        flags = [getattr(self, name) for name in _VALIDITY_FIELDS]
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("synthetic receipt flags must be booleans")
        blockers = tuple(self.blockers)
        warnings = tuple(self.warnings)
        if any(not isinstance(item, LearningIssue) for item in (*blockers, *warnings)):
            raise TypeError("synthetic receipt findings are invalid")
        passing = (
            all(flags) and all(run.status == "pass" for run in runs) and not blockers
        )
        if (self.status == "pass") != passing:
            raise ValueError("synthetic receipt status does not match evidence")
        if tuple(self.claims_made) != CLAIMS or len(set(self.claims_made)) != len(
            self.claims_made
        ):
            raise ValueError("synthetic receipt claims are invalid")
        if not set(NON_CLAIMS).issubset(self.claims_not_made) or len(
            set(self.claims_not_made)
        ) != len(self.claims_not_made):
            raise ValueError("synthetic receipt non-claims are invalid")
        if not isinstance(self.metadata, Mapping) or "parameters" in self.metadata:
            raise ValueError("synthetic receipt metadata is invalid")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(sorted(self.metadata.items())))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "problem_id": self.problem_id,
            "architecture_id": self.architecture_id,
            "optimizer_id": self.optimizer_id,
            "whole_student": self.whole_student.to_dict(),
            "trunk_only": self.trunk_only.to_dict(),
            "head_only": self.head_only.to_dict(),
            "resume": self.resume.to_dict(),
            **{name: getattr(self, name) for name in _VALIDITY_FIELDS},
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_made": list(self.claims_made),
            "claims_not_made": list(self.claims_not_made),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class P39SmokeDependencies:
    run_loop_fn: Callable[..., LearningLoopResult] = run_learning_loop
    checkpoint_write_fn: Callable[[LearningCheckpoint, Path], LearningCheckpoint] = (
        save_learning_checkpoint
    )
    checkpoint_restore_fn: Callable[[Path], LearningCheckpoint] = (
        load_learning_checkpoint
    )
    build_report_fn: Callable[..., LearningRunReport] = build_learning_run_report
    temporary_directory_factory: Callable[[], str] = tempfile.mkdtemp


@dataclass(frozen=True)
class _SyntheticArchitecture:
    architecture_id: str = ARCHITECTURE_ID
    architecture_version: int = 1

    def capability_profile(self):
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            1,
            (
                "architecture.batch_validation_v1",
                "architecture.forward_v1",
                "architecture.initialize_parameters_v1",
                "architecture.objective.final_output_v1",
                "architecture.parameter_metadata_v1",
                "architecture.update_scope.named_region_v1",
                "architecture.update_scope.whole_student_v1",
            ),
        )

    def validate_config(self, config):
        if config.architecture_id != self.architecture_id:
            raise ValueError("synthetic architecture configuration mismatch")

    def describe_parameters(self, parameters=None):
        del parameters
        return ParameterCatalog(
            self.architecture_id,
            (
                ParameterDescriptor(
                    "trunk.weight",
                    (),
                    "float64",
                    "recurrent_block",
                    ("trunk", "whole_student"),
                ),
                ParameterDescriptor(
                    "head.bias", (), "float64", "output_head", ("head", "whole_student")
                ),
            ),
        )

    def architecture_metadata(self):
        catalog = self.describe_parameters()
        return ArchitectureMetadata(
            self.architecture_id,
            catalog,
            self.capability_profile(),
            (
                NamedRegion("whole_student", catalog.paths),
                NamedRegion("trunk", ("trunk.weight",)),
                NamedRegion("head", ("head.bias",)),
            ),
            (
                IntermediateSurfaceDescriptor(
                    "final_output",
                    "scalar",
                    available_in_training=True,
                    available_in_inference=True,
                ),
            ),
        )

    def initialize_parameters(self, request):
        self.validate_config(request.config)
        return ArchitectureInitResult(
            self.describe_parameters(),
            ArchitectureState("synthetic_linear.initial"),
            {"trunk.weight": 0.0, "head.bias": 0.0},
        )

    def validate_batch(self, batch, config):
        self.validate_config(config)
        x, y = batch.inputs.get("x"), batch.targets.get("y")
        if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y) == 5:
            return BatchValidationResult("pass")
        raise ValueError("synthetic batch must contain five x/y values")

    def forward(self, request):
        x = request.batch.inputs["x"]
        return ForwardResult(
            outputs=[
                request.parameters["trunk.weight"] * item
                + request.parameters["head.bias"]
                for item in x
            ],
            intermediate_surfaces=("final_output",),
        )

    def resolve_update_scope(self, scope, catalog):
        if scope.kind == "whole_student":
            selected = catalog.paths
        elif scope.kind == "named_region" and scope.region_id in {"trunk", "head"}:
            selected = (
                self.architecture_metadata().region(scope.region_id).parameter_paths
            )
        else:
            raise ValueError("synthetic update scope is unsupported")
        return ResolvedUpdateSelection(
            "synthetic:" + ",".join(selected),
            tuple(selected),
            tuple(path for path in catalog.paths if path not in selected),
        )

    def resolve_objective_scope(self, scope, metadata):
        if (
            scope.kind != "final_output"
            or metadata.architecture_id != self.architecture_id
        ):
            raise ValueError("synthetic objective scope is unsupported")
        return ResolvedObjectiveSelection(scope, "final_output")


@dataclass(frozen=True)
class _MseObjective:
    def evaluate(self, parameters, batch):
        xs, ys = batch.inputs["x"], batch.targets["y"]
        errors = [
            parameters["trunk.weight"] * x + parameters["head.bias"] - y
            for x, y in zip(xs, ys, strict=True)
        ]
        count = len(errors)
        return sum(error * error for error in errors) / count, {
            "trunk.weight": 2.0
            * sum(error * x for error, x in zip(errors, xs, strict=True))
            / count,
            "head.bias": 2.0 * sum(errors) / count,
        }


class _ObservationHook:
    hook_id = "p3_9_observer"
    priority = 0
    supported_events = ("loop_start", "step_end", "checkpoint", "loop_end")

    def __init__(self):
        self.events: list[tuple[str, int, int]] = []

    def on_event(self, context: HookContext) -> HookResult:
        self.events.append(
            (context.event_type, context.event_sequence, context.global_step)
        )
        return HookResult(
            metrics=(MetricRecord("synthetic.hook_observed", 1.0, context.global_step),)
        )


@dataclass(frozen=True)
class _RunEvidence:
    summary: SyntheticRunSummary
    result: LearningLoopResult
    parameters: Mapping[str, float]
    optimizer_state: Any
    learning_state: LearningState
    source_state: Mapping[str, object]
    hook_events: tuple[tuple[str, int, int], ...]


def _batch() -> LearningBatch:
    return LearningBatch(
        "synthetic:y=2x+1", {"x": [-2, -1, 0, 1, 2]}, {"y": [-3, -1, 1, 3, 5]}
    )


def _initial(scope: str, run_id: str):
    architecture = _SyntheticArchitecture()
    config = ArchitectureConfig(ARCHITECTURE_ID)
    catalog = architecture.describe_parameters()
    update_scope = (
        UpdateScope()
        if scope == "whole_student"
        else UpdateScope("named_region", scope)
    )
    optimizer = SgdOptimizer()
    optimizer_config = OptimizerConfig(OPTIMIZER_ID, learning_rate=0.1)
    selection = architecture.resolve_update_scope(update_scope, catalog)
    optimizer_state = optimizer.initialize_state(
        OptimizerInitRequest(optimizer_config, catalog, selection)
    ).optimizer_state
    learning_state = LearningState(run_id, active_update_scope=update_scope)
    return (
        architecture,
        config,
        optimizer,
        optimizer_config,
        optimizer_state,
        learning_state,
        {"trunk.weight": 0.0, "head.bias": 0.0},
    )


def _loss(parameters: Mapping[str, float]) -> float:
    return _MseObjective().evaluate(parameters, _batch())[0]


def _execute(
    *,
    mode: str,
    scope: str,
    steps: int,
    deps: P39SmokeDependencies,
    source: SyntheticBatchSource | None = None,
    state: tuple[Any, ...] | None = None,
    checkpoint: Callable[[Any], str] | None = None,
    run_id: str | None = None,
) -> _RunEvidence:
    if state is None:
        (
            architecture,
            config,
            optimizer,
            optimizer_config,
            optimizer_state,
            learning_state,
            parameters,
        ) = _initial(scope, run_id or f"p3_9:{mode}")
    else:
        (
            architecture,
            config,
            optimizer,
            optimizer_config,
            optimizer_state,
            learning_state,
            parameters,
        ) = state
    initial_loss = _loss(parameters)
    source = source or SyntheticBatchSource(
        (_batch(),) * (steps + 2), source_id="p3_9.synthetic"
    )
    hook = _ObservationHook()
    result = deps.run_loop_fn(
        config=LearningLoopConfig(
            steps, checkpoint_every_n_steps=3 if checkpoint else None
        ),
        architecture=architecture,
        architecture_config=config,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        learning_state=learning_state,
        parameters=parameters,
        objective=_MseObjective(),
        batch_source=source,
        checkpoint=checkpoint,
        hooks=(hook,),
        emit_run_report=True,
    )
    execution = result.final_execution
    final_parameters = execution.parameters if execution else parameters
    final_optimizer_state = execution.optimizer_state if execution else optimizer_state
    final_learning_state = execution.learning_state if execution else learning_state
    # Exercise the public opt-in attachment and independently validate its pure
    # builder seam, which makes report corruption observable to the smoke gate.
    report = deps.build_report_fn(
        loop_result=result,
        run_id=final_learning_state.run_id,
        update_scope=final_learning_state.active_update_scope.kind,
        objective_scope="final_output",
    )
    result = replace(result, report=report)
    deltas = {path: final_parameters[path] - parameters[path] for path in parameters}
    changed = tuple(path for path in sorted(deltas) if deltas[path] != 0.0)
    summary = SyntheticRunSummary(
        final_learning_state.run_id,
        mode,
        result.status,
        result.stop_reason,
        result.steps_completed,
        result.global_step,
        initial_loss,
        _loss(final_parameters),
        _loss(final_parameters) / initial_loss,
        deltas,
        changed,
        tuple(path for path in sorted(deltas) if path not in changed),
        len(result.checkpoints),
        report.schema_version if report else None,
    )
    return _RunEvidence(
        summary,
        result,
        dict(final_parameters),
        final_optimizer_state,
        final_learning_state,
        dict(source.state_dict()),
        tuple(hook.events),
    )


def _write_checkpoint(
    evidence: _RunEvidence, directory: Path, deps: P39SmokeDependencies
) -> str:
    checkpoint = LearningCheckpoint(
        "p3_9.runtime",
        evidence.learning_state,
        ArchitectureState("synthetic_linear.state"),
        evidence.optimizer_state,
        evidence.parameters,
        {},
        {},
    )
    deps.checkpoint_write_fn(checkpoint, directory)
    (directory / "source.json").write_text(
        json.dumps(evidence.source_state, sort_keys=True, separators=(",", ":"))
    )
    return "p3_9_checkpoint_" + str(evidence.learning_state.global_step)


def _restore_checkpoint(directory: Path, deps: P39SmokeDependencies, scope: str):
    checkpoint = deps.checkpoint_restore_fn(directory, runtime_reference="p3_9.runtime")
    source_path = directory / "source.json"
    if not source_path.is_file():
        raise ValueError("checkpoint source state is missing")
    source_state = json.loads(source_path.read_text())
    architecture, config, optimizer, optimizer_config, _, _, _ = _initial(
        scope, checkpoint.learning_state.run_id
    )
    if (
        checkpoint.architecture_state is None
        or checkpoint.architecture_state.state_id != "synthetic_linear.state"
    ):
        raise ValueError("checkpoint architecture mismatch")
    if checkpoint.optimizer_state.optimizer_id != OPTIMIZER_ID:
        raise ValueError("checkpoint optimizer mismatch")
    source = SyntheticBatchSource((_batch(),) * 16, source_id="p3_9.synthetic")
    source.load_state_dict(source_state)
    return (
        architecture,
        config,
        optimizer,
        optimizer_config,
        checkpoint.optimizer_state,
        checkpoint.learning_state,
        checkpoint.parameters,
    ), source


def _issue(code: str, section: str, check: str, expected: Any, actual: Any):
    from radjax_student.learning import LearningIssue

    return LearningIssue(
        code,
        "P3.9 synthetic learning smoke failed",
        {"section": section, "check": check, "expected": expected, "actual": actual},
    )


def _default_dependencies() -> P39SmokeDependencies:
    return P39SmokeDependencies()


def run_p3_9_synthetic_learning_smoke(
    dependencies: P39SmokeDependencies | None = None,
) -> P39SyntheticLearningReceipt:
    deps = dependencies or _default_dependencies()
    blockers: list[object] = []
    values = {name: False for name in _VALIDITY_FIELDS}
    failed = SyntheticRunSummary(
        "p3_9:unavailable",
        "whole_student",
        "fail",
        "setup_failed",
        0,
        0,
        1.0,
        1.0,
        1.0,
        {"head.bias": 0.0, "trunk.weight": 0.0},
        (),
        ("head.bias", "trunk.weight"),
        0,
        None,
    )
    whole = trunk = head = resume = failed
    try:
        whole_evidence = _execute(
            mode="whole_student",
            scope="whole_student",
            steps=12,
            deps=deps,
            checkpoint=lambda execution: "checkpoint",
        )
        whole = whole_evidence.summary
        values["loss_decrease_valid"] = (
            whole.final_loss <= whole.initial_loss * 0.5
            and whole.changed_parameter_paths == ("head.bias", "trunk.weight")
        )
        if not values["loss_decrease_valid"]:
            blockers.append(
                _issue(
                    "p3_9_loss_threshold_failed",
                    "whole_student",
                    "loss_and_movement",
                    "loss <= initial * 0.5 and both parameters changed",
                    whole.to_dict(),
                )
            )
        required_metrics = {
            "loss",
            "gradient_norm",
            "parameter_norm",
            "learning_rate",
            "changed_parameter_count",
            "unchanged_parameter_count",
        }
        values["metrics_valid"] = required_metrics.issubset(
            {item.name for item in whole_evidence.result.metrics}
        ) and "synthetic.hook_observed" in {
            item.name for item in whole_evidence.result.metrics
        }
        if not values["metrics_valid"]:
            blockers.append(
                _issue(
                    "p3_9_metrics_missing",
                    "metrics",
                    "required_metrics",
                    sorted(required_metrics),
                    sorted({item.name for item in whole_evidence.result.metrics}),
                )
            )
        # The loop interleaves checkpoint events after steps 3, 6, 9, and 12.
        expected_events = (
            ["loop_start"]
            + [
                event
                for step in range(1, 13)
                for event in (
                    ["step_end", "checkpoint"] if step % 3 == 0 else ["step_end"]
                )
            ]
            + ["loop_end"]
        )
        values["hooks_valid"] = [
            event[0] for event in whole_evidence.hook_events
        ] == expected_events and all(
            event in whole_evidence.result.hook_events for event in expected_events
        )
        if not values["hooks_valid"]:
            blockers.append(
                _issue(
                    "p3_9_hook_observation_failed",
                    "hooks",
                    "event_order",
                    expected_events,
                    [event[0] for event in whole_evidence.hook_events],
                )
            )
        values["run_reporting_valid"] = (
            whole_evidence.result.report is not None
            and whole_evidence.result.report.schema_version == REPORT_SCHEMA
            and whole_evidence.result.report.scopes.update_scope == "whole_student"
            and whole_evidence.result.report.status.global_step == 12
        )
        if not values["run_reporting_valid"]:
            blockers.append(
                _issue(
                    "p3_9_run_report_failed",
                    "run_reporting",
                    "whole_report",
                    REPORT_SCHEMA,
                    None
                    if whole_evidence.result.report is None
                    else whole_evidence.result.report.to_dict(),
                )
            )

        trunk_evidence = _execute(mode="trunk_only", scope="trunk", steps=6, deps=deps)
        head_evidence = _execute(mode="head_only", scope="head", steps=6, deps=deps)
        trunk, head = trunk_evidence.summary, head_evidence.summary
        values["scope_boundaries_valid"] = (
            trunk.changed_parameter_paths == ("trunk.weight",)
            and head.changed_parameter_paths == ("head.bias",)
            and trunk_evidence.result.report.scopes.update_scope == "named_region"
            and head_evidence.result.report.scopes.update_scope == "named_region"
        )
        if not values["scope_boundaries_valid"]:
            blockers.append(
                _issue(
                    "p3_9_scope_boundary_failed",
                    "scopes",
                    "excluded_parameters",
                    {"trunk": ["trunk.weight"], "head": ["head.bias"]},
                    {
                        "trunk": list(trunk.changed_parameter_paths),
                        "head": list(head.changed_parameter_paths),
                    },
                )
            )
        values["optimizer_boundaries_valid"] = (
            trunk_evidence.optimizer_state.backend_state["per_parameter_steps"]
            == {"head.bias": 0, "trunk.weight": 6}
            and head_evidence.optimizer_state.backend_state["per_parameter_steps"]
            == {"head.bias": 6, "trunk.weight": 0}
        )
        if not values["optimizer_boundaries_valid"]:
            blockers.append(
                _issue(
                    "p3_9_optimizer_boundary_failed",
                    "optimizer",
                    "per_parameter_state",
                    "excluded state remains zero",
                    {
                        "trunk": trunk_evidence.optimizer_state.backend_state,
                        "head": head_evidence.optimizer_state.backend_state,
                    },
                )
            )

        temp_dir = Path(deps.temporary_directory_factory())
        try:
            uninterrupted = _execute(
                mode="resume", scope="whole_student", steps=12, deps=deps
            )
            first = _execute(mode="resume", scope="whole_student", steps=6, deps=deps)
            _write_checkpoint(first, temp_dir, deps)
            restored_state, restored_source = _restore_checkpoint(
                temp_dir, deps, "whole_student"
            )
            second = _execute(
                mode="resume",
                scope="whole_student",
                steps=6,
                deps=deps,
                source=restored_source,
                state=restored_state,
            )
            resume = replace(
                second.summary,
                steps_completed=first.summary.steps_completed
                + second.summary.steps_completed,
                checkpoint_count=1,
            )
            values["checkpoint_restore_valid"] = (
                second.parameters == uninterrupted.parameters
                and second.optimizer_state.backend_state
                == uninterrupted.optimizer_state.backend_state
                and second.learning_state == uninterrupted.learning_state
                and restored_source.state_dict()["position"] == 12
            )
            if not values["checkpoint_restore_valid"]:
                blockers.append(
                    _issue(
                        "p3_9_resume_mismatch",
                        "checkpoint_resume",
                        "restored_continuation",
                        uninterrupted.parameters,
                        second.parameters,
                    )
                )
            corrupt = temp_dir / "architecture.json"
            corrupt.write_text("{}")
            try:
                _restore_checkpoint(temp_dir, deps, "whole_student")
            except ValueError:
                corruption_rejected = True
            else:
                corruption_rejected = False
            if not corruption_rejected:
                values["checkpoint_restore_valid"] = False
                blockers.append(
                    _issue(
                        "p3_9_checkpoint_corruption_not_detected",
                        "checkpoint",
                        "corrupt_hash",
                        "ValueError",
                        "accepted",
                    )
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        replay = _execute(
            mode="replay",
            scope="whole_student",
            steps=12,
            deps=deps,
            checkpoint=lambda execution: "checkpoint",
            run_id=whole_evidence.learning_state.run_id,
        )
        values["deterministic_replay_valid"] = (
            replay.parameters == whole_evidence.parameters
            and replay.optimizer_state.backend_state
            == whole_evidence.optimizer_state.backend_state
            and replay.result.metrics == whole_evidence.result.metrics
            and replay.hook_events == whole_evidence.hook_events
            and replay.result.report.to_dict() == whole_evidence.result.report.to_dict()
        )
        if not values["deterministic_replay_valid"]:
            blockers.append(
                _issue(
                    "p3_9_replay_mismatch",
                    "replay",
                    "exact_equality",
                    whole_evidence.summary.to_dict(),
                    replay.summary.to_dict(),
                )
            )
    except Exception as exc:
        blockers.append(
            _issue(
                "p3_9_internal_error",
                "execution",
                "unexpected_exception",
                "no exception",
                type(exc).__name__,
            )
        )
    status = (
        "pass"
        if all(values.values())
        and all(run.status == "pass" for run in (whole, trunk, head, resume))
        and not blockers
        else "fail"
    )
    return P39SyntheticLearningReceipt(
        SCHEMA,
        status,
        PROBLEM_ID,
        ARCHITECTURE_ID,
        OPTIMIZER_ID,
        whole,
        trunk,
        head,
        resume,
        blockers=tuple(blockers),
        metadata={"numeric_tolerance": SYNTHETIC_NUMERIC_TOLERANCE, "offline": True},
        **values,
    )


def _outcome(valid: bool) -> str:
    return "pass" if valid else "fail"


def main(
    argv: list[str] | None = None, dependencies: P39SmokeDependencies | None = None
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_p3_9_synthetic_learning_smoke(dependencies)
    if args.json:
        print(receipt.to_json())
    else:
        print("P3.9 Synthetic Learning Smoke")
        print(f"  status: {receipt.status.upper()}")
        print("  problem: y = 2x + 1")
        print("  architecture: synthetic_linear_v1")
        print("  optimizer: sgd")
        print(
            f"  whole-student loss decreased: {_outcome(receipt.loss_decrease_valid)}"
        )
        print(f"  trunk-only boundary: {_outcome(receipt.scope_boundaries_valid)}")
        print(f"  head-only boundary: {_outcome(receipt.scope_boundaries_valid)}")
        print(f"  checkpoint resume: {_outcome(receipt.checkpoint_restore_valid)}")
        print(f"  deterministic replay: {_outcome(receipt.deterministic_replay_valid)}")
        metrics_and_hooks = receipt.metrics_valid and receipt.hooks_valid
        print(f"  metrics and hooks: {_outcome(metrics_and_hooks)}")
        print(f"  run reporting: {_outcome(receipt.run_reporting_valid)}")
        print(f"  blockers: {len(receipt.blockers)}")
        print(f"  warnings: {len(receipt.warnings)}")
        print("  note: synthetic systems proof, not a model-quality claim")
    return 0 if receipt.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
