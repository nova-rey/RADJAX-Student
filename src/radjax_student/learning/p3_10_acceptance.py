"""P3.10.1 public-seam golden acceptance gate for the learning core."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

from radjax_student import architecture as _architecture
from radjax_student.checkpoints import (
    LearningCheckpoint,
    load_learning_checkpoint,
    save_learning_checkpoint,
)
from radjax_student.checkpoints.learning import CHECKPOINT_SCHEMA_VERSION
from radjax_student.learning import (
    LearningBatch,
    LearningConfig,
    LearningContractError,
    LearningIssue,
    LearningState,
    MetricRecord,
    ObjectiveRequest,
    ObjectiveResult,
    ObjectiveScope,
    UpdateScope,
    canonical_learning_json,
    canonical_objective_json,
    run_p3_8_observability_acceptance,
)
from radjax_student.learning.models import CheckpointPolicy
from radjax_student.learning.scopes import ResolvedUpdateSelection
from radjax_student.learning.synthetic_smoke import (
    P39SyntheticLearningReceipt,
    run_p3_9_synthetic_learning_smoke,
)
from radjax_student.optimizers import (
    GradientTree,
    OptimizerConfig,
    OptimizerContractError,
    OptimizerInitRequest,
    OptimizerState,
    OptimizerUpdateRequest,
    SgdOptimizer,
)
from radjax_student.steps.loop import (
    LearningLoopConfig,
    SyntheticBatchSource,
    run_learning_loop,
)
from radjax_student.steps.single import learning_step

ArchitectureCapabilityProfile = _architecture.ArchitectureCapabilityProfile
ArchitectureConfig = _architecture.ArchitectureConfig
ArchitectureInitRequest = _architecture.ArchitectureInitRequest
ArchitectureInitResult = _architecture.ArchitectureInitResult
ArchitectureIssue = _architecture.ArchitectureIssue
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

SCHEMA = "radjax.p3_10_learning_core_acceptance.v1"
CLAIMS = (
    "p3_1_learning_contract_validated",
    "p3_2_architecture_boundary_validated",
    "p3_3_optimizer_validated",
    "p3_4_batch_objective_validated",
    "p3_5_single_step_validated",
    "p3_6_checkpoint_validated",
    "p3_7_learning_loop_validated",
    "p3_8_observability_validated",
    "p3_9_synthetic_learning_validated",
    "p3_10_learning_core_golden_gate_validated",
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
VALIDITY_FIELDS = (
    "contracts_valid",
    "optimizer_valid",
    "single_step_valid",
    "loop_valid",
    "checkpoint_valid",
    "resume_valid",
    "observability_valid",
    "synthetic_learning_valid",
    "deterministic_replay_valid",
    "documentation_valid",
    "test_inventory_valid",
)
SECTION_CODES = {
    "contracts_valid": "p3_10_contracts_failed",
    "optimizer_valid": "p3_10_optimizer_failed",
    "single_step_valid": "p3_10_single_step_failed",
    "loop_valid": "p3_10_loop_failed",
    "checkpoint_valid": "p3_10_checkpoint_failed",
    "resume_valid": "p3_10_resume_failed",
    "observability_valid": "p3_10_observability_failed",
    "synthetic_learning_valid": "p3_10_synthetic_learning_failed",
    "deterministic_replay_valid": "p3_10_deterministic_replay_failed",
    "documentation_valid": "p3_10_documentation_failed",
    "test_inventory_valid": "p3_10_test_inventory_failed",
}


@dataclass(frozen=True)
class P310AcceptanceDependencies:
    architecture_factory: Callable[[], Any]
    optimizer_factory: Callable[[], Any]
    single_step_fn: Callable[..., Any]
    run_loop_fn: Callable[..., Any]
    checkpoint_save_fn: Callable[..., Any]
    checkpoint_load_fn: Callable[..., Any]
    observability_acceptance_fn: Callable[[], Any]
    synthetic_smoke_fn: Callable[[], P39SyntheticLearningReceipt]
    source_loader: Callable[[Path], str]
    path_exists_fn: Callable[[Path], bool]
    temporary_directory_factory: Callable[[], str]


@dataclass(frozen=True)
class P310LearningCoreAcceptanceReceipt:
    schema_version: str
    status: Literal["pass", "fail"]
    contracts_valid: bool
    optimizer_valid: bool
    single_step_valid: bool
    loop_valid: bool
    checkpoint_valid: bool
    resume_valid: bool
    observability_valid: bool
    synthetic_learning_valid: bool
    deterministic_replay_valid: bool
    documentation_valid: bool
    test_inventory_valid: bool
    blockers: tuple[LearningIssue, ...] = ()
    warnings: tuple[LearningIssue, ...] = ()
    claims_made: tuple[str, ...] = CLAIMS
    claims_not_made: tuple[str, ...] = NON_CLAIMS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA or self.status not in {"pass", "fail"}:
            raise ValueError("invalid P3.10 acceptance receipt")
        flags = {name: getattr(self, name) for name in VALIDITY_FIELDS}
        if any(type(value) is not bool for value in flags.values()):
            raise TypeError("P3.10 validity flags must be booleans")
        blockers = tuple(self.blockers)
        warnings = tuple(self.warnings)
        if any(
            not isinstance(issue, LearningIssue) for issue in (*blockers, *warnings)
        ):
            raise TypeError("P3.10 findings must be LearningIssue values")
        if (self.status == "pass") != (all(flags.values()) and not blockers):
            raise ValueError("P3.10 status does not match evidence")
        if self.claims_made != CLAIMS or len(set(self.claims_made)) != len(CLAIMS):
            raise ValueError("P3.10 claims are invalid")
        if not set(NON_CLAIMS).issubset(self.claims_not_made):
            raise ValueError("P3.10 non-claims are incomplete")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("P3.10 metadata must be a mapping")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            **{name: getattr(self, name) for name in VALIDITY_FIELDS},
            "blockers": [issue.to_dict() for issue in self.blockers],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "claims_made": list(self.claims_made),
            "claims_not_made": list(self.claims_not_made),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class _GoldenArchitecture:
    architecture_id = "p310.synthetic_linear.v1"
    architecture_version = 1

    def capability_profile(self):
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            self.architecture_version,
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
            raise ValueError("architecture configuration mismatch")

    def describe_parameters(self, parameters=None):
        del parameters
        return ParameterCatalog(
            self.architecture_id,
            (
                ParameterDescriptor(
                    "head.bias", (), "float64", "output_head", ("head", "whole_student")
                ),
                ParameterDescriptor(
                    "trunk.weight",
                    (),
                    "float64",
                    "recurrent_block",
                    ("trunk", "whole_student"),
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
                NamedRegion("head", ("head.bias",)),
                NamedRegion("trunk", ("trunk.weight",)),
                NamedRegion("whole_student", catalog.paths),
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
        catalog = self.describe_parameters()
        return ArchitectureInitResult(
            catalog,
            ArchitectureState("p310.synthetic.initial"),
            {"head.bias": 0.0, "trunk.weight": 0.0},
        )

    def validate_batch(self, batch, config):
        self.validate_config(config)
        values = batch.inputs.get("x"), batch.targets.get("y")
        if all(isinstance(value, tuple) and len(value) == 5 for value in values):
            return BatchValidationResult("pass")
        return BatchValidationResult(
            "fail",
            (ArchitectureIssue("p310_batch_invalid", "expected five x/y values"),),
        )

    def forward(self, request):
        x_values = request.batch.inputs["x"]
        weight = request.parameters["trunk.weight"]
        bias = request.parameters["head.bias"]
        return ForwardResult(
            outputs=tuple(weight * x + bias for x in x_values),
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
            raise ValueError("unsupported synthetic update scope")
        return ResolvedUpdateSelection(
            f"p310:{scope.kind}:{','.join(selected)}",
            selected,
            tuple(path for path in catalog.paths if path not in selected),
        )

    def resolve_objective_scope(self, scope, metadata):
        if (
            scope.kind != "final_output"
            or metadata.architecture_id != self.architecture_id
        ):
            raise ValueError("unsupported synthetic objective scope")
        return ResolvedObjectiveSelection(scope, "final_output")


class _GoldenObjective:
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


def _batch() -> LearningBatch:
    return LearningBatch(
        "p310:y=2x+1",
        {"x": (-2, -1, 0, 1, 2)},
        {"y": (-3, -1, 1, 3, 5)},
    )


def _setup(deps: P310AcceptanceDependencies, scope: str = "whole_student"):
    architecture = deps.architecture_factory()
    architecture_config = ArchitectureConfig(architecture.architecture_id)
    catalog = architecture.describe_parameters()
    update_scope = (
        UpdateScope()
        if scope == "whole_student"
        else UpdateScope("named_region", scope)
    )
    selection = architecture.resolve_update_scope(update_scope, catalog)
    optimizer = deps.optimizer_factory()
    optimizer_config = OptimizerConfig(optimizer.optimizer_id, learning_rate=0.1)
    optimizer_state = optimizer.initialize_state(
        OptimizerInitRequest(optimizer_config, catalog, selection)
    ).optimizer_state
    learning_state = LearningState(f"p310:{scope}", active_update_scope=update_scope)
    parameters = {path: 0.0 for path in catalog.paths}
    return (
        architecture,
        architecture_config,
        optimizer,
        optimizer_config,
        optimizer_state,
        learning_state,
        parameters,
    )


def _objective_result(parameters, batch):
    return _GoldenObjective().evaluate(parameters, batch)


def _audit_contracts(deps: P310AcceptanceDependencies) -> bool:
    architecture = deps.architecture_factory()
    config = LearningConfig(
        update_scope=UpdateScope(), objective_scope=ObjectiveScope()
    )
    batch = _batch()
    state = LearningState("p310-contract")
    request = ObjectiveRequest("p310-objective", batch_reference=batch.batch_id)
    objective = ObjectiveResult("p310-objective", 1.0, {"mse": 1.0})
    policy = CheckpointPolicy("every_n_steps", every_n_steps=2)
    metric = MetricRecord("loss", 1.0, 0)
    models = (
        config,
        batch,
        state,
        request,
        objective,
        UpdateScope(),
        ObjectiveScope(),
        policy,
        metric,
    )
    round_trips = [type(model).from_dict(model.to_dict()).to_dict() for model in models]
    architecture_ok = (
        architecture.architecture_id == "p310.synthetic_linear.v1"
        and architecture.capability_profile().supports("architecture.forward_v1")
        and architecture.describe_parameters().paths == ("head.bias", "trunk.weight")
        and architecture.resolve_update_scope(
            UpdateScope(), architecture.describe_parameters()
        ).selected_parameter_paths
        == ("head.bias", "trunk.weight")
        and architecture.resolve_update_scope(
            UpdateScope("named_region", "trunk"), architecture.describe_parameters()
        ).selected_parameter_paths
        == ("trunk.weight",)
        and architecture.resolve_update_scope(
            UpdateScope("named_region", "head"), architecture.describe_parameters()
        ).selected_parameter_paths
        == ("head.bias",)
        and architecture.resolve_objective_scope(
            ObjectiveScope(), architecture.architecture_metadata()
        ).surface_id
        == "final_output"
    )
    config = ArchitectureConfig(architecture.architecture_id)
    initialized = architecture.initialize_parameters(
        ArchitectureInitRequest(config, "p310.runtime")
    )
    valid_batch = architecture.validate_batch(_batch(), config).ok
    invalid_batch = LearningBatch("p310:invalid", {"x": (0,)}, {"y": (1,)})
    invalid_batch_rejected = not architecture.validate_batch(invalid_batch, config).ok
    forward = architecture.forward(
        ForwardRequest(
            _batch(),
            parameters={"trunk.weight": 2.0, "head.bias": 1.0},
            training=True,
        )
    )
    invalid_scope_rejected = False
    try:
        architecture.resolve_update_scope(
            UpdateScope("named_region", "missing"), architecture.describe_parameters()
        )
    except (OptimizerContractError, TypeError, ValueError):
        invalid_scope_rejected = True
    invalid_objective_rejected = False
    try:
        architecture.resolve_objective_scope(
            ObjectiveScope("intermediate_surface", "missing"),
            architecture.architecture_metadata(),
        )
    except (OptimizerContractError, TypeError, ValueError):
        invalid_objective_rejected = True
    config_mismatch_rejected = False
    try:
        architecture.validate_config(ArchitectureConfig("wrong"))
    except (OptimizerContractError, TypeError, ValueError):
        config_mismatch_rejected = True
    invalid_rejected = 0
    for factory in (
        lambda: LearningConfig(max_steps=-1),
        lambda: UpdateScope("invalid"),
        lambda: ObjectiveScope("intermediate_surface"),
        lambda: CheckpointPolicy("every_n_steps"),
        lambda: MetricRecord("loss", float("nan"), 0),
        lambda: ObjectiveResult("loss", float("inf")),
    ):
        try:
            factory()
        except (LearningContractError, TypeError, ValueError):
            invalid_rejected += 1
    return (
        all(
            left == right
            for left, right in zip(
                (m.to_dict() for m in models), round_trips, strict=True
            )
        )
        and canonical_learning_json(config.to_dict())
        == canonical_learning_json(config.to_dict())
        and canonical_objective_json(request.to_dict())
        == canonical_objective_json(request.to_dict())
        and architecture_ok
        and initialized.parameter_catalog.paths == ("head.bias", "trunk.weight")
        and valid_batch
        and invalid_batch_rejected
        and forward.outputs == (-3.0, -1.0, 1.0, 3.0, 5.0)
        and "final_output" in forward.intermediate_surfaces
        and invalid_scope_rejected
        and invalid_objective_rejected
        and config_mismatch_rejected
        and invalid_rejected == 6
    )


def _audit_optimizer(deps: P310AcceptanceDependencies) -> bool:
    _, _, optimizer, optimizer_config, state, _, parameters = _setup(deps)
    catalog = deps.architecture_factory().describe_parameters()
    selection = ResolvedUpdateSelection("p310:trunk", ("trunk.weight",), ("head.bias",))
    result = optimizer.apply_updates(
        OptimizerUpdateRequest(
            GradientTree(catalog.paths, values={path: 1.0 for path in catalog.paths}),
            state,
            optimizer_config,
            selection,
            0,
            parameters=parameters,
            schedule_values={"learning_rate": 0.25},
        )
    )
    replay = optimizer.apply_updates(
        OptimizerUpdateRequest(
            GradientTree(catalog.paths, values={path: 1.0 for path in catalog.paths}),
            state,
            optimizer_config,
            selection,
            0,
            parameters=parameters,
            schedule_values={"learning_rate": 0.25},
        )
    )
    invalid = 0
    invalid_builders = (
        lambda: OptimizerUpdateRequest(
            GradientTree(("unknown",), values={"unknown": 1.0}),
            state,
            optimizer_config,
            selection,
            0,
            parameters=parameters,
        ),
        lambda: OptimizerUpdateRequest(
            GradientTree(catalog.paths, values={path: 1.0 for path in catalog.paths}),
            OptimizerState("wrong", state.parameter_paths),
            optimizer_config,
            selection,
            0,
            parameters=parameters,
        ),
    )
    for build_request in invalid_builders:
        try:
            optimizer.apply_updates(build_request())
        except (TypeError, ValueError):
            invalid += 1
    nonfinite_rejected = False
    try:
        optimizer.apply_updates(
            OptimizerUpdateRequest(
                GradientTree(
                    catalog.paths,
                    values={"head.bias": float("nan"), "trunk.weight": 1.0},
                ),
                state,
                optimizer_config,
                selection,
                0,
                parameters=parameters,
            )
        )
    except (OptimizerContractError, TypeError, ValueError):
        nonfinite_rejected = True
    mismatch_request = SimpleNamespace(
        gradients=GradientTree(
            catalog.paths, values={path: 1.0 for path in catalog.paths}
        ),
        optimizer_state=state,
        config=OptimizerConfig("wrong.optimizer", learning_rate=0.25),
        resolved_update_selection=selection,
        learning_step=0,
        parameters=parameters,
        schedule_values={"learning_rate": 0.25},
    )
    mismatch_rejected = False
    try:
        optimizer.apply_updates(mismatch_request)
    except (OptimizerContractError, TypeError, ValueError):
        mismatch_rejected = True
    result_steps = result.updated_optimizer_state.backend_state["per_parameter_steps"]
    replay_steps = replay.updated_optimizer_state.backend_state["per_parameter_steps"]
    return (
        optimizer.optimizer_id == "sgd.v1"
        and state.step == 0
        and result.updated_optimizer_state.step == 1
        and result.updated_parameters["trunk.weight"] == -0.25
        and result.updated_parameters["head.bias"] == 0.0
        and result_steps["trunk.weight"] == 1
        and result_steps["head.bias"] == 0
        and result.update_metadata["learning_rate"] == 0.25
        and replay.updated_parameters == result.updated_parameters
        and replay.updated_optimizer_state.to_dict()
        == result.updated_optimizer_state.to_dict()
        and replay_steps == result_steps
        and replay.parameter_updates == result.parameter_updates
        and invalid == 2
        and nonfinite_rejected
        and mismatch_rejected
    )


def _audit_single_step(deps: P310AcceptanceDependencies) -> bool:
    setup = _setup(deps, scope="trunk")
    (
        architecture,
        architecture_config,
        optimizer,
        optimizer_config,
        state,
        learning_state,
        parameters,
    ) = setup
    execution = deps.single_step_fn(
        batch=_batch(),
        architecture=architecture,
        architecture_config=architecture_config,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=state,
        learning_state=learning_state,
        parameters=parameters,
        objective=_GoldenObjective(),
    )
    expected_loss, gradients = _objective_result(parameters, _batch())
    expected_delta = -0.1 * gradients["trunk.weight"]
    failure_unchanged = False
    invalid_gradient_rejected = False
    nonfinite_loss_rejected = False

    class FailingObjective:
        def evaluate(self, parameters, batch):
            raise RuntimeError("p310 objective failure")

    try:
        deps.single_step_fn(
            batch=_batch(),
            architecture=architecture,
            architecture_config=architecture_config,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            optimizer_state=state,
            learning_state=learning_state,
            parameters=parameters,
            objective=FailingObjective(),
        )
    except Exception:
        failure_unchanged = (
            parameters == {"head.bias": 0.0, "trunk.weight": 0.0}
            and state.step == 0
            and learning_state.global_step == 0
        )

    class InvalidGradientObjective:
        def evaluate(self, parameters, batch):
            del parameters, batch
            return 1.0, {"unknown.path": 1.0, "trunk.weight": 1.0}

    try:
        deps.single_step_fn(
            batch=_batch(),
            architecture=architecture,
            architecture_config=architecture_config,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            optimizer_state=state,
            learning_state=learning_state,
            parameters=parameters,
            objective=InvalidGradientObjective(),
        )
    except (OptimizerContractError, TypeError, ValueError):
        invalid_gradient_rejected = True

    class NonFiniteLossObjective:
        def evaluate(self, parameters, batch):
            del parameters, batch
            return float("nan"), {"trunk.weight": 1.0, "head.bias": 0.0}

    try:
        deps.single_step_fn(
            batch=_batch(),
            architecture=architecture,
            architecture_config=architecture_config,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            optimizer_state=state,
            learning_state=learning_state,
            parameters=parameters,
            objective=NonFiniteLossObjective(),
        )
    except (TypeError, ValueError):
        nonfinite_loss_rejected = True
    return (
        execution.result.status == "pass"
        and execution.result.global_step_before == 0
        and execution.result.global_step_after == 1
        and execution.learning_state.global_step == 1
        and execution.optimizer_state.step == 1
        and execution.result.loss.loss == expected_loss
        and execution.parameters["trunk.weight"] == expected_delta
        and execution.result.changed_parameter_paths == ("trunk.weight",)
        and execution.result.unchanged_parameter_paths == ("head.bias",)
        and execution.parameters["head.bias"] == parameters["head.bias"] == 0.0
        and execution.optimizer_state.backend_state["per_parameter_steps"]["head.bias"]
        == state.backend_state["per_parameter_steps"]["head.bias"]
        and {metric.name for metric in execution.result.metrics}
        >= {"loss", "gradient_norm"}
        and failure_unchanged
        and invalid_gradient_rejected
        and nonfinite_loss_rejected
    )


def _audit_loop(deps: P310AcceptanceDependencies) -> bool:
    setup = _setup(deps)
    (
        architecture,
        architecture_config,
        optimizer,
        optimizer_config,
        state,
        learning_state,
        parameters,
    ) = setup
    source = SyntheticBatchSource((_batch(),) * 4, source_id="p310.loop")
    checkpoints: list[str] = []
    result = deps.run_loop_fn(
        config=LearningLoopConfig(4, checkpoint_every_n_steps=2),
        architecture=architecture,
        architecture_config=architecture_config,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=state,
        learning_state=learning_state,
        parameters=parameters,
        objective=_GoldenObjective(),
        batch_source=source,
        checkpoint=lambda execution: (
            checkpoints.append(str(execution.learning_state.global_step))
            or checkpoints[-1]
        ),
    )
    exhausted_source = SyntheticBatchSource((_batch(),) * 2, source_id="p310.exhausted")
    exhausted = deps.run_loop_fn(
        config=LearningLoopConfig(4),
        architecture=architecture,
        architecture_config=architecture_config,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=state,
        learning_state=learning_state,
        parameters=parameters,
        objective=_GoldenObjective(),
        batch_source=exhausted_source,
    )

    class FailingObjective:
        calls = 0

        def evaluate(self, parameters, batch):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("p310 loop failure")
            return _GoldenObjective().evaluate(parameters, batch)

    failure_source = SyntheticBatchSource((_batch(),) * 4, source_id="p310.failure")
    failure = deps.run_loop_fn(
        config=LearningLoopConfig(4, checkpoint_every_n_steps=2),
        architecture=architecture,
        architecture_config=architecture_config,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=state,
        learning_state=learning_state,
        parameters=parameters,
        objective=FailingObjective(),
        batch_source=failure_source,
        checkpoint=lambda execution: str(execution.learning_state.global_step),
    )
    resumed_setup = _setup(deps)
    resumed = deps.run_loop_fn(
        config=LearningLoopConfig(2),
        architecture=resumed_setup[0],
        architecture_config=resumed_setup[1],
        optimizer=resumed_setup[2],
        optimizer_config=resumed_setup[3],
        optimizer_state=replace(resumed_setup[4], step=10),
        learning_state=replace(resumed_setup[5], global_step=10, optimizer_step=10),
        parameters=resumed_setup[6],
        objective=_GoldenObjective(),
        batch_source=SyntheticBatchSource((_batch(),) * 2, source_id="p310.resumed"),
    )
    return (
        result.status == "pass"
        and result.stop_reason == "max_steps"
        and result.steps_completed == result.global_step == result.batches_consumed == 4
        and result.final_execution is not None
        and source.position == 4
        and checkpoints == ["2", "4"]
        and result.checkpoints == ("2", "4")
        and bool(result.metrics)
        and exhausted.stop_reason == "source_exhausted"
        and exhausted.steps_completed == exhausted.global_step == 2
        and failure.stop_reason == "learning_step_failure"
        and failure.steps_completed == failure.global_step == 1
        and not failure.checkpoints
        and resumed.global_step == 12
        and resumed.steps_completed == 2
    )


def _audit_checkpoint(deps: P310AcceptanceDependencies) -> bool:
    state = OptimizerState(
        "sgd.v1", ("p310.weight",), step=1, backend_state={"steps": 1}
    )
    checkpoint = LearningCheckpoint(
        "p310-runtime",
        LearningState("p310", global_step=1, optimizer_step=1),
        ArchitectureState("p310.arch"),
        state,
        {"p310.weight": 0.9},
        {"source_id": "p310.source", "position": 1},
        {},
        {},
    )
    path = Path(deps.temporary_directory_factory())
    saved = deps.checkpoint_save_fn(checkpoint, path)
    loaded = deps.checkpoint_load_fn(path, runtime_reference="p310-runtime")
    manifest = json.loads((path / "manifest.json").read_text())
    files_ok = all(
        (path / name).is_file()
        for name in (
            "architecture.json",
            "learning.json",
            "optimizer.json",
            "source.json",
            "manifest.json",
        )
    )

    def rewrite_manifest(target: Path, mutate) -> None:
        payload = json.loads((target / "manifest.json").read_text())
        payload.pop("integrity", None)
        mutate(payload)
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        payload["integrity"] = {
            "algorithm": "sha256",
            "manifest_digest": hashlib.sha256(encoded).hexdigest(),
        }
        (target / "manifest.json").write_bytes(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )

    def rejected(target: Path) -> bool:
        try:
            deps.checkpoint_load_fn(target)
        except Exception:
            return True
        return False

    tamper_checks = []
    tamper_mutations = (
        lambda target: (target / "source.json").write_text("{}"),
        lambda target: (target / "source.json").unlink(),
        lambda target: (target / "manifest.json").write_text("{}"),
        lambda target: rewrite_manifest(
            target, lambda item: item["hashes"].update({"source.json": "0" * 64})
        ),
        lambda target: rewrite_manifest(
            target, lambda item: item["sizes"].update({"source.json": 0})
        ),
        lambda target: rewrite_manifest(
            target, lambda item: item["ownership"].update({"source.json": "wrong"})
        ),
        lambda target: rewrite_manifest(
            target, lambda item: item.update({"schema_version": "unsupported"})
        ),
    )
    for index, mutate in enumerate(tamper_mutations):
        target = path / f"tamper-{index}"
        deps.checkpoint_save_fn(checkpoint, target)
        mutate(target)
        tamper_checks.append(rejected(target))

    altered_target = path / "tamper-altered-source-state"
    deps.checkpoint_save_fn(checkpoint, altered_target)
    altered_source = json.loads((altered_target / "source.json").read_text())
    altered_source["source_state"]["position"] = 99
    altered_bytes = (
        json.dumps(altered_source, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (altered_target / "source.json").write_bytes(altered_bytes)
    rewrite_manifest(
        altered_target,
        lambda item: (
            item["hashes"].update(
                {"source.json": hashlib.sha256(altered_bytes).hexdigest()}
            ),
            item["sizes"].update({"source.json": len(altered_bytes)}),
        ),
    )
    try:
        altered_loaded = deps.checkpoint_load_fn(altered_target)
    except Exception:
        altered_state_rejected = True
    else:
        altered_state_rejected = altered_loaded.source_state != checkpoint.source_state

    none_target = path / "none-source-state"
    deps.checkpoint_save_fn(replace(checkpoint, source_state=None), none_target)
    none_loaded = deps.checkpoint_load_fn(none_target)
    try:
        deps.checkpoint_load_fn(path, runtime_reference="wrong")
    except ValueError:
        runtime_rejected = True
    else:
        runtime_rejected = False
    return (
        saved.schema_version == CHECKPOINT_SCHEMA_VERSION == "learning_checkpoint.v2"
        and loaded.source_state["position"] == 1
        and loaded.parameters == checkpoint.parameters
        and loaded.learning_state == checkpoint.learning_state
        and loaded.optimizer_state.backend_state
        == checkpoint.optimizer_state.backend_state
        and loaded.architecture_state == checkpoint.architecture_state
        and files_ok
        and all(tamper_checks)
        and altered_state_rejected
        and none_loaded.source_state is None
        and runtime_rejected
        and manifest["ownership"]["source.json"] == "batch_source"
        and "source.json" in manifest["hashes"]
        and "source.json" in manifest["sizes"]
    )


def _audit_resume(deps: P310AcceptanceDependencies) -> bool:
    receipt = deps.synthetic_smoke_fn()
    return (
        receipt.status == "pass"
        and receipt.checkpoint_restore_valid
        and receipt.resume.status == "pass"
        and receipt.resume.global_step == receipt.whole_student.global_step
        and receipt.resume.report_schema_version == "radjax.learning_run_report.v1"
        and not receipt.blockers
    )


def _audit_observability(deps: P310AcceptanceDependencies) -> bool:
    receipt = deps.observability_acceptance_fn()
    fields = tuple(name for name in dir(receipt) if name.endswith("_valid"))
    return (
        receipt.status == "pass"
        and bool(fields)
        and not receipt.blockers
        and receipt.to_json() == receipt.to_json()
    )


def _audit_synthetic(deps: P310AcceptanceDependencies) -> bool:
    receipt = deps.synthetic_smoke_fn()
    return (
        receipt.status == "pass"
        and receipt.loss_decrease_valid
        and receipt.scope_boundaries_valid
        and receipt.optimizer_boundaries_valid
        and receipt.checkpoint_restore_valid
        and receipt.metrics_valid
        and receipt.hooks_valid
        and receipt.run_reporting_valid
        and receipt.deterministic_replay_valid
        and not receipt.blockers
    )


def _source_text(deps: P310AcceptanceDependencies, path: Path) -> str:
    return deps.source_loader(path)


def _documentation_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "README.md",
        root / "bible.md",
        root / "docs/INDEX.md",
        *(
            root / "docs" / f"P3_{number}_{name}.md"
            for number, name in (
                ("1", "GENERIC_LEARNING_CONTRACT"),
                ("2", "STUDENT_ARCHITECTURE_PLUGIN_CONTRACT"),
                ("3", "OPTIMIZER_CONTRACT"),
                ("4", "GENERIC_BATCH_AND_OBJECTIVE_CONTRACT"),
                ("5", "SINGLE_LEARNING_STEP"),
                ("6", "MODEL_AND_OPTIMIZER_CHECKPOINT_CONTRACT"),
                ("7", "GENERIC_LEARNING_LOOP"),
                ("8", "METRICS_HOOKS_AND_REPORTING"),
                ("9", "SYNTHETIC_END_TO_END_LEARNING_SMOKE"),
            )
        ),
        root / "docs/P3_8D_OBSERVABILITY_GOLDEN_ACCEPTANCE_GATE.md",
        root / "docs/P3_10_LEARNING_CORE_GOLDEN_ACCEPTANCE.md",
    )


def _audit_documentation(deps: P310AcceptanceDependencies) -> bool:
    root = Path(__file__).parents[3]
    paths = _documentation_paths(root)
    if not all(deps.path_exists_fn(path) for path in paths):
        return False
    readme, bible, index = (
        _source_text(deps, root / name)
        for name in ("README.md", "bible.md", "docs/INDEX.md")
    )
    p36 = _source_text(
        deps, root / "docs/P3_6_MODEL_AND_OPTIMIZER_CHECKPOINT_CONTRACT.md"
    )
    p38 = _source_text(
        deps, root / "docs/P3_8D_OBSERVABILITY_GOLDEN_ACCEPTANCE_GATE.md"
    )
    p39 = _source_text(deps, root / "docs/P3_9_SYNTHETIC_END_TO_END_LEARNING_SMOKE.md")
    p310 = _source_text(deps, root / "docs/P3_10_LEARNING_CORE_GOLDEN_ACCEPTANCE.md")
    return (
        "learning_checkpoint.v2" in p36
        and "batch_source" in p36
        and "P3.8D" in p38
        and "y = 2x + 1" in p39
        and "source.json" in p39
        and "independently" in p310.lower()
        and "P3.10" in readme
        and "P3.10" in index
        and "Phase 3" in bible
        and "closure" in bible.lower()
    )


def _phase3_test_paths(root: Path) -> tuple[Path, ...]:
    names = (
        "test_learning_contract.py",
        "test_architecture_plugin_contract.py",
        "test_batch_objective_contract.py",
        "test_optimizer_contract.py",
        "test_single_learning_step.py",
        "test_learning_checkpoint.py",
        "test_learning_loop.py",
        "test_p3_8_observability_acceptance.py",
        "test_p3_9_synthetic_learning_smoke.py",
        "test_p3_10_learning_core_acceptance.py",
    )
    return tuple(root / "tests" / name for name in names)


def _audit_inventory(deps: P310AcceptanceDependencies) -> bool:
    root = Path(__file__).parents[3]
    paths = _phase3_test_paths(root)
    if not all(deps.path_exists_fn(path) for path in paths):
        return False
    total = 0
    for path in paths:
        tree = ast.parse(_source_text(deps, path), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                if node.name in names:
                    return False
                names.add(node.name)
                total += 1
                if any(isinstance(item, ast.Pass) for item in ast.walk(node)):
                    return False
            if (
                isinstance(node, ast.Assert)
                and isinstance(node.test, ast.Constant)
                and node.test.value is True
            ):
                return False
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "skip"
            ):
                return False
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("tests.")
                and any(alias.name.startswith("test_") for alias in node.names)
            ):
                return False
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.startswith("test_")
            ):
                return False
    return total >= 120


def _snapshot(deps: P310AcceptanceDependencies) -> str:
    architecture = deps.architecture_factory()
    optimizer = deps.optimizer_factory()
    contract = _audit_contracts(deps)
    optimizer_ok = _audit_optimizer(deps)
    single_ok = _audit_single_step(deps)
    loop_ok = _audit_loop(deps)
    checkpoint_ok = _audit_checkpoint(deps)
    observability = deps.observability_acceptance_fn().to_json()
    synthetic = deps.synthetic_smoke_fn().to_json()
    payload = {
        "architecture": architecture.architecture_metadata().to_dict(),
        "optimizer": optimizer.optimizer_id,
        "contracts": contract,
        "optimizer_valid": optimizer_ok,
        "single_step_valid": single_ok,
        "loop_valid": loop_ok,
        "checkpoint_valid": checkpoint_ok,
        "observability": observability,
        "synthetic": synthetic,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _audit_replay(deps: P310AcceptanceDependencies) -> bool:
    first, second = _snapshot(deps), _snapshot(deps)
    return first == second


def _default_dependencies() -> P310AcceptanceDependencies:
    return P310AcceptanceDependencies(
        architecture_factory=_GoldenArchitecture,
        optimizer_factory=SgdOptimizer,
        single_step_fn=learning_step,
        run_loop_fn=run_learning_loop,
        checkpoint_save_fn=save_learning_checkpoint,
        checkpoint_load_fn=load_learning_checkpoint,
        observability_acceptance_fn=run_p3_8_observability_acceptance,
        synthetic_smoke_fn=run_p3_9_synthetic_learning_smoke,
        source_loader=lambda path: path.read_text(encoding="utf-8"),
        path_exists_fn=lambda path: path.is_file(),
        temporary_directory_factory=tempfile.mkdtemp,
    )


def _finding(code: str, section: str, expected: Any, actual: Any) -> LearningIssue:
    return LearningIssue(
        code,
        "P3.10.1 learning core acceptance section failed",
        {"section": section, "check": section, "expected": expected, "actual": actual},
    )


def run_p3_10_learning_core_acceptance(
    dependencies: P310AcceptanceDependencies | None = None,
) -> P310LearningCoreAcceptanceReceipt:
    deps = _default_dependencies() if dependencies is None else dependencies
    audits = (
        ("contracts_valid", _audit_contracts),
        ("optimizer_valid", _audit_optimizer),
        ("single_step_valid", _audit_single_step),
        ("loop_valid", _audit_loop),
        ("checkpoint_valid", _audit_checkpoint),
        ("resume_valid", _audit_resume),
        ("observability_valid", _audit_observability),
        ("synthetic_learning_valid", _audit_synthetic),
        ("deterministic_replay_valid", _audit_replay),
        ("documentation_valid", _audit_documentation),
        ("test_inventory_valid", _audit_inventory),
    )
    values: dict[str, bool] = {}
    blockers: list[LearningIssue] = []
    for field_name, audit in audits:
        try:
            values[field_name] = bool(audit(deps))
        except Exception as exc:
            values[field_name] = False
            blockers.append(
                _finding(
                    "p3_10_internal_error",
                    field_name,
                    "no exception",
                    type(exc).__name__,
                )
            )
        if not values[field_name]:
            blockers.append(
                _finding(SECTION_CODES[field_name], field_name, True, False)
            )
    return P310LearningCoreAcceptanceReceipt(
        schema_version=SCHEMA,
        status="pass" if all(values.values()) and not blockers else "fail",
        blockers=tuple(blockers),
        metadata={
            "gate": "P3.10.1",
            "section_count": len(audits),
            "independent_seams": True,
        },
        **values,
    )


def main(
    argv: list[str] | None = None,
    dependencies: P310AcceptanceDependencies | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_p3_10_learning_core_acceptance(dependencies)
    if args.json:
        print(receipt.to_json())
    else:
        passed = sum(getattr(receipt, name) for name in VALIDITY_FIELDS)
        print("P3.10 Learning Core Golden Acceptance")
        print(f"  status: {receipt.status.upper()}")
        print(f"  independent sections: {passed}/{len(VALIDITY_FIELDS)}")
        print(f"  blockers: {len(receipt.blockers)}")
        print("  note: seam acceptance evidence, not model quality")
    return 0 if receipt.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
