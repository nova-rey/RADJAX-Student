"""Validation-only execution of the accepted stateful P3.11.8 conveyor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from radjax_student.architecture import (
    ArchitectureCapabilityProfile,
    ArchitectureConfig,
    ArchitectureInitRequest,
    ArchitectureInitResult,
    ArchitectureMetadata,
    ArchitectureRegistry,
    ArchitectureState,
    BatchValidationResult,
    ForwardResult,
    IntermediateSurfaceDescriptor,
    NamedRegion,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.architecture.testing import (
    FAKE_ARCHITECTURE_CAPABILITIES,
    FakeArchitecturePlugin,
)
from radjax_student.checkpoints import save_learning_checkpoint_v3
from radjax_student.checkpoints.npz_codec import (
    describe_mapping_pytree,
    descriptor_digest,
    mapping_pytree_digest,
)
from radjax_student.contracts import (
    HFArchitectureProjection,
    HFCompatibilityDescriptor,
    HFParameterProjection,
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    ObjectiveConfig,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.learning import (
    HookResult,
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningBatch,
    LearningState,
    MetricRecord,
    ObjectiveScope,
    UpdateScope,
    assemble_jax_learning_lifecycle,
)
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import (
    OptimizerConfig,
    OptimizerRegistry,
    SgdOptimizer,
)
from radjax_student.runtime import (
    RuntimeConfig,
    build_default_runtime_registry,
)
from radjax_student.steps import (
    JaxLearningLifecycle,
    JaxLoopExecutor,
    LearningLoopConfig,
    SyntheticBatchSource,
    run_learning_loop,
)
from radjax_student.validation.p3_11_9_replay.canonical import (
    canonical_digest,
    canonical_metric_mapping,
    finite_float_hex,
    parse_finite_float_hex,
)
from radjax_student.validation.p3_11_9_replay.models import (
    ArchitectureCarryIdentityEvidence,
    CrossModeComparisonEvidence,
    ExperimentIdentityEvidence,
    HFPreservationEvidence,
    ObjectiveEvidence,
    OptimizerConfigEvidence,
    ReplayArmEvidence,
    ReplayRunEvidence,
    ReplayStepEvidence,
    RngEvidence,
    RuntimeEvidence,
    StatefulReplayProof,
    ToleranceEvidence,
)

ARCHITECTURE_ID = "validation.stateful_linear_jax.v1"
NON_CLAIMS = (
    "no_production_architecture",
    "no_tome_payload_consumption",
    "no_distillation",
    "no_hf_export",
    "no_accelerator_scale_training",
    "no_multi_device_proof",
    "no_cross_hardware_bitwise_determinism",
    "no_cross_version_bitwise_determinism",
    "no_performance_claim",
    "no_radlads_parity_claim",
    "no_phase4_readiness_before_p3_11_10",
)


def _digest(value: Any) -> str:
    return canonical_digest(value)


class StatefulLinearJaxArchitecture(FakeArchitecturePlugin):
    """Validation-only complete plugin; never a production architecture."""

    def capability_profile(self) -> ArchitectureCapabilityProfile:
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            self.architecture_version,
            (*FAKE_ARCHITECTURE_CAPABILITIES, "architecture.jax_execution_v1"),
        )

    def describe_parameters(self, parameters=None) -> ParameterCatalog:
        del parameters
        return ParameterCatalog(
            self.architecture_id,
            (
                ParameterDescriptor(
                    "trunk.weight",
                    (1,),
                    "float32",
                    "recurrent_block",
                    ("trunk", "whole_student"),
                ),
                ParameterDescriptor(
                    "head.bias",
                    (1,),
                    "float32",
                    "output_head",
                    ("head", "whole_student"),
                ),
            ),
        )

    def architecture_metadata(self) -> ArchitectureMetadata:
        catalog = self.describe_parameters()
        return ArchitectureMetadata(
            self.architecture_id,
            catalog,
            self.capability_profile(),
            named_regions=(
                NamedRegion("trunk", ("trunk.weight",)),
                NamedRegion("head", ("head.bias",)),
                NamedRegion("whole_student", catalog.paths),
            ),
            objective_surfaces=(
                IntermediateSurfaceDescriptor(
                    "final_output", "regression", available_in_training=True
                ),
            ),
        )

    def validate_batch(self, batch, config):
        self.validate_config(config)
        if not isinstance(batch.inputs.get("x"), (list, tuple)) or not isinstance(
            batch.targets.get("y"), (list, tuple)
        ):
            return BatchValidationResult("fail")
        return BatchValidationResult("pass")

    def initialize_parameters(
        self, request: ArchitectureInitRequest
    ) -> ArchitectureInitResult:
        self.validate_config(request.config)
        layout = _layout()
        carry = {
            "forwards": jnp.asarray(0, dtype=jnp.int32),
            "rng_probe": jnp.asarray(0.0, dtype=jnp.float32),
        }
        catalog = self.describe_parameters()
        descriptor = self._hf_descriptor(request, catalog, layout)
        return ArchitectureInitResult(
            parameter_catalog=catalog,
            architecture_state=ArchitectureState("stateful-linear-state.v1"),
            parameters={
                "trunk": {"weight": jnp.asarray((0.0,), dtype=jnp.float32)},
                "head": {"bias": jnp.asarray((0.0,), dtype=jnp.float32)},
            },
            architecture_carry=carry,
            architecture_carry_descriptor={
                "schema_version": "architecture_carry.v1",
                "state_id": "stateful-linear-state.v1",
                "pytree_descriptor_digest": descriptor_digest(
                    describe_mapping_pytree(carry)
                ),
            },
            parameter_layout=layout,
            hf_descriptor=descriptor,
        )

    def _hf_descriptor(self, request, catalog, layout):
        projections = tuple(
            HFParameterProjection(
                logical_path=entry.logical_path,
                jax_keypath=entry.jax_keypath,
                shape=entry.shape,
                dtype=entry.dtype,
                exportability="exportable" if entry.exportable else "non_exportable",
                hf_distribution_key=entry.hf_distribution_key,
                projection_rule="identity",
                tied_parameter_group=entry.tied_weight_group,
                non_exportability_reason=(
                    None if entry.exportable else "validation_fixture_not_exported"
                ),
            )
            for entry in layout.entries
        )
        return HFCompatibilityDescriptor(
            schema_version="hf_compatibility_descriptor.v2",
            architecture_id=self.architecture_id,
            architecture_plugin_version=self.architecture_version,
            model_type="stateful_linear_validation",
            architecture_config_digest=_digest(request.config.to_dict()),
            parameter_catalog_digest=_digest(catalog.to_dict()),
            parameter_layout_digest=layout.digest(),
            tokenizer=HFTokenizerIdentity(
                "stateful-test-tokenizer",
                "synthetic-r1",
                _digest({"tokenizer": "stateful-linear"}),
                _digest({"tokenizer_config": "stateful-linear"}),
                "synthetic_wordlevel",
                _digest({"normalization": "identity"}),
                "synthetic",
            ),
            vocabulary=HFVocabularyIdentity(
                8,
                _digest({"vocabulary": list(range(8))}),
                _digest({"token_to_id": list(range(8))}),
                _digest({"added_tokens": []}),
                "0-3",
            ),
            special_tokens=HFSpecialTokenIdentity(0, 1, 2, 3, None, ()),
            parameter_projections=projections,
            architecture_projection=HFArchitectureProjection(
                "stateful_linear_config",
                "stateful_linear_validation",
                1,
                1,
                8,
                request.config.sequence_length or 1,
                {"dtype_intent": request.config.dtype_intent},
            ),
            non_claims=("no_hf_export", "validation_only_architecture"),
            notes="Synthetic validation descriptor; no Hugging Face export is claimed.",
        )

    def hf_compatibility_descriptor(self, request, result):
        return self._hf_descriptor(
            request, result.parameter_catalog, result.parameter_layout
        )

    def apply_jax(
        self,
        parameters,
        architecture_state,
        batch,
        *,
        objective_scope,
        training,
        rng_key,
    ):
        del objective_scope, training
        probe = jax.random.uniform(rng_key, (), dtype=jnp.float32)
        output = (
            batch.inputs["x"][:, None] * parameters["trunk"]["weight"]
            + parameters["head"]["bias"]
        )
        return ForwardResult(
            outputs=output,
            updated_architecture_carry={
                "forwards": architecture_state["forwards"] + 1,
                "rng_probe": probe,
            },
            architecture_metrics={"forward_count": architecture_state["forwards"]},
        )


class EventHook:
    hook_id = "p3119.events"
    priority = 0
    supported_events = (
        "loop_start",
        "batch_received",
        "step_start",
        "step_end",
        "checkpoint",
        "loop_end",
        "failure",
    )

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.failures: list[dict[str, Any]] = []

    def on_event(self, context):
        self.events.append((context.event_type, context.global_step))
        if context.event_type == "failure":
            self.failures.append(dict(context.metadata))
        return HookResult(
            metrics=(
                MetricRecord(
                    "hook_events", float(context.event_sequence), context.global_step
                ),
            )
        )


def _layout() -> ParameterTreeLayout:
    return ParameterTreeLayout(
        ARCHITECTURE_ID,
        (
            ParameterTreeLayoutEntry(
                "trunk.weight",
                ("trunk", "weight"),
                (1,),
                "float32",
                "recurrent_block",
                ("trunk", "whole_student"),
            ),
            ParameterTreeLayoutEntry(
                "head.bias",
                ("head", "bias"),
                (1,),
                "float32",
                "output_head",
                ("head", "whole_student"),
            ),
        ),
    )


def _batch(index: int) -> LearningBatch:
    # A one-element dyadic regression avoids backend-dependent reduction and
    # fused-arithmetic rounding while still exercising the complete conveyor.
    return LearningBatch(
        f"stateful-{index}",
        inputs={"x": [1.0]},
        targets={"y": [[2.0]]},
    )


def _new_assembly(mode: str, objects: list[object]):
    """Validation invokes the production assembly authority, never a local recipe."""
    architecture_registry = ArchitectureRegistry()
    architecture = StatefulLinearJaxArchitecture(architecture_id=ARCHITECTURE_ID)
    architecture_registry.register(architecture)
    architecture = architecture_registry.get(ARCHITECTURE_ID)
    config = ArchitectureConfig(ARCHITECTURE_ID, vocab_size=8, dtype_intent="float32")
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(SgdOptimizer())
    optimizer = optimizer_registry.get("sgd.v1")
    optimizer_config = OptimizerConfig("sgd.v1", learning_rate=0.25)
    objective_registry = build_default_objective_registry()
    objective_config = ObjectiveConfig(
        CANONICAL_MSE_IDENTITY,
        {"reduction": "mean"},
    )
    runtime_config = RuntimeConfig(
        backend_id="jax",
        platform_preference="cpu",
        precision_policy="float32",
        placement_policy="single_device",
        compilation_policy=mode,
        distributed_policy="disabled",
        fallback_policy="disallowed",
        seed=17,
    )
    registry = build_default_runtime_registry()
    request = JaxLearningAssemblyRequest(
        architecture_id=ARCHITECTURE_ID,
        architecture_version=architecture.architecture_version,
        architecture_config=config,
        objective_identity=CANONICAL_MSE_IDENTITY,
        objective_config=objective_config,
        optimizer_id="sgd.v1",
        optimizer_version=optimizer.optimizer_version,
        optimizer_config=optimizer_config,
        runtime_backend_id="jax",
        runtime_implementation_version="p2.9",
        runtime_config=runtime_config,
        root_seed=17,
        learning_state=LearningState(
            "p3119",
            active_update_scope=UpdateScope("named_region", "trunk"),
            active_objective_scope=ObjectiveScope(),
        ),
    )
    assembled = assemble_jax_learning_lifecycle(
        request,
        registries=JaxLearningAssemblyRegistries(
            architecture_registry, objective_registry, optimizer_registry, registry
        ),
    )
    objects.extend(
        (
            architecture_registry,
            architecture,
            optimizer_registry,
            optimizer,
            objective_registry,
            registry,
            assembled,
        )
    )
    return assembled


def _new_lifecycle(mode: str, objects: list[object]) -> JaxLearningLifecycle:
    """Compatibility projection for historical validators.

    This is intentionally not a second construction recipe: P3.11.9, P3.12A,
    and P3.12B all receive the lifecycle produced by the one learning-owned
    production assembler above.
    """

    return _new_assembly(mode, objects).lifecycle


class _RecordingExecutor:
    """Validation wrapper; the underlying executor remains the production path."""

    def __init__(
        self,
        inner: JaxLoopExecutor,
        hook: EventHook,
        records: list[tuple[LearningBatch, Any]],
    ) -> None:
        self.inner = inner
        self.hook = hook
        self.records = records

    def __call__(self, **kwargs: Any):
        lifecycle = self.inner.lifecycle
        # The resumed arm intentionally swaps in a freshly initialized lifecycle.
        kwargs.update(
            architecture=lifecycle.architecture,
            architecture_config=lifecycle.architecture_config,
            optimizer=lifecycle.optimizer,
            optimizer_config=lifecycle.optimizer_config,
            optimizer_state=lifecycle.optimizer_state,
            learning_state=lifecycle.learning_state,
            parameters=lifecycle.parameters,
        )
        execution = self.inner(**kwargs)
        self.records.append((kwargs["batch"], execution))
        return execution


def _runtime_evidence(execution) -> RuntimeEvidence:
    receipt = execution.runtime_result
    metadata = dict(receipt.output_metadata)
    preparation = dict(metadata["input_preparation"])
    preparation.pop("selected_device_id", None)
    return RuntimeEvidence(
        backend_id=receipt.backend_id,
        mode=receipt.mode,
        compiled=receipt.compiled,
        dispatched=receipt.dispatched,
        synchronized=receipt.synchronized,
        placement_policy=preparation["placement_policy"],
        precision_policy=preparation["precision_policy"],
        output_metadata_fields=tuple(sorted(metadata)),
        non_claims=tuple(receipt.claims_not_made),
    )


def _objective_evidence(lifecycle: JaxLearningLifecycle) -> ObjectiveEvidence:
    descriptor = lifecycle.objective_descriptor
    return ObjectiveEvidence(
        objective_id=descriptor.identity.objective_id,
        objective_version=descriptor.identity.objective_version,
        capability_profile_digest=descriptor.capability_profile_digest,
        config_digest=descriptor.config_digest,
        resolved_surface_identity=descriptor.resolved_surface_identity,
        metric_schema_id=descriptor.metric_schema_id,
        implementation_identity=descriptor.implementation_identity,
        descriptor_digest=descriptor.digest,
    )


def _step_evidence(
    records: list[tuple[LearningBatch, Any]], hook: EventHook
) -> tuple[ReplayStepEvidence, ...]:
    result: list[ReplayStepEvidence] = []
    hook_values = tuple(f"{event}:{step}" for event, step in hook.events)
    for index, (batch, execution) in enumerate(records):
        state = execution.learning_state
        step_result = execution.result
        result.append(
            ReplayStepEvidence(
                step_index=index,
                batch_id=batch.batch_id,
                batch_digest=_digest(batch.to_dict()),
                objective=_objective_evidence_from_descriptor(
                    execution.objective_descriptor
                ),
                update_scope_digest=_digest(step_result.active_update_scope.to_dict()),
                counters_before={
                    "global_step": step_result.global_step_before,
                    "micro_step": 0,
                    "optimizer_step": step_result.global_step_before,
                },
                counters_after={
                    "global_step": state.global_step,
                    "micro_step": state.micro_step,
                    "optimizer_step": state.optimizer_step,
                },
                parameter_digest=mapping_pytree_digest(execution.parameters),
                architecture_carry_digest=mapping_pytree_digest(
                    execution.architecture_carry
                ),
                optimizer_array_digest=mapping_pytree_digest(
                    execution.optimizer_state.arrays
                ),
                optimizer_envelope_digest=_digest(
                    execution.optimizer_state.envelope.to_dict()
                ),
                changed_paths=step_result.changed_parameter_paths,
                unchanged_paths=step_result.unchanged_parameter_paths,
                objective_metrics=canonical_metric_mapping(execution.objective_metrics),
                architecture_metrics=canonical_metric_mapping(
                    execution.architecture_metrics
                ),
                optimizer_metrics=canonical_metric_mapping(execution.optimizer_metrics),
                hook_events=tuple(
                    value
                    for value in hook_values
                    if value.endswith(f":{state.global_step}")
                    or value.endswith(f":{step_result.global_step_before}")
                ),
                runtime=_runtime_evidence(execution),
                rng=RngEvidence.from_dict(
                    execution.runtime_result.output_metadata["rng_bridge"]
                ),
            )
        )
    return tuple(result)


def _identity(lifecycle: JaxLearningLifecycle) -> ExperimentIdentityEvidence:
    return ExperimentIdentityEvidence(
        architecture_id=lifecycle.architecture.architecture_id,
        architecture_config_digest=lifecycle.config_digest,
        parameter_catalog_digest=lifecycle.catalog_digest,
        parameter_layout_digest=lifecycle.parameter_layout.digest(),
        hf_reference=HFPreservationEvidence(
            descriptor_schema_version=lifecycle.hf_descriptor.schema_version,
            descriptor_digest=lifecycle.hf_descriptor.digest,
            architecture_id=lifecycle.hf_descriptor.architecture_id,
            model_type=lifecycle.hf_descriptor.model_type,
            architecture_config_digest=(
                lifecycle.hf_descriptor.architecture_config_digest
            ),
            parameter_catalog_digest=(lifecycle.hf_descriptor.parameter_catalog_digest),
            parameter_layout_digest=(lifecycle.hf_descriptor.parameter_layout_digest),
            parameter_projection_digest=(
                lifecycle.hf_descriptor.parameter_projection_digest
            ),
            architecture_projection_digest=(
                lifecycle.hf_descriptor.architecture_projection.digest
            ),
            tokenizer_identity_digest=lifecycle.hf_descriptor.tokenizer.digest,
            vocabulary_identity_digest=lifecycle.hf_descriptor.vocabulary.digest,
            special_token_identity_digest=(
                lifecycle.hf_descriptor.special_tokens.digest
            ),
            preservation_reference_digest=lifecycle.hf_reference.digest,
        ),
        objective=_objective_evidence(lifecycle),
        architecture_state_id=(
            None
            if lifecycle.architecture_state is None
            else lifecycle.architecture_state.state_id
        ),
        architecture_carry_descriptor=ArchitectureCarryIdentityEvidence.from_dict(
            lifecycle.architecture_carry_descriptor
        ),
        optimizer_id=lifecycle.optimizer.optimizer_id,
        optimizer_capability_version=lifecycle.optimizer.optimizer_version,
        optimizer_numerical_state_schema_version=(
            lifecycle.optimizer_state.descriptor.optimizer_schema_version
        ),
        optimizer_config=OptimizerConfigEvidence.from_dict(
            lifecycle.optimizer_config.to_dict()
        ),
        runtime_reference=lifecycle.runtime_reference,
        root_seed=lifecycle.runtime_context.root_seed,
    )


def _objective_evidence_from_descriptor(descriptor) -> ObjectiveEvidence:
    return ObjectiveEvidence(
        objective_id=descriptor.identity.objective_id,
        objective_version=descriptor.identity.objective_version,
        capability_profile_digest=descriptor.capability_profile_digest,
        config_digest=descriptor.config_digest,
        resolved_surface_identity=descriptor.resolved_surface_identity,
        metric_schema_id=descriptor.metric_schema_id,
        implementation_identity=descriptor.implementation_identity,
        descriptor_digest=descriptor.digest,
    )


@dataclass(frozen=True)
class _ExecutedArm:
    """Ephemeral comparison inputs; raw arrays never leave validation execution."""

    evidence: ReplayArmEvidence
    parameters: Any
    architecture_carry: Any
    optimizer_arrays: Any
    learning_state: Mapping[str, Any]
    optimizer_envelope: Mapping[str, Any]
    hook_events: tuple[tuple[str, int], ...]
    retained_metrics: tuple[tuple[str, Any], ...]
    logical_paths: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    rng_coordinates: tuple[RngEvidence, ...]
    runtimes: tuple[RuntimeEvidence, ...]
    final_runtime: RuntimeEvidence
    report: Any
    loop_result: Any


def _run_arm(
    mode: str, arm: str, directory: Path, objects: list[object]
) -> _ExecutedArm:
    assembled = _new_assembly(mode, objects)
    lifecycle = assembled.lifecycle
    initial_identity = _identity(lifecycle)
    hook = EventHook()
    records: list[tuple[LearningBatch, Any]] = []
    inner = assembled.loop_executor
    executor = _RecordingExecutor(inner, hook, records)
    source = SyntheticBatchSource(tuple(_batch(index) for index in range(6)))
    objects.extend((hook, inner, executor, source))
    checkpoint_manifest_digest: str | None = None
    restored = False

    def checkpoint(_execution):
        nonlocal checkpoint_manifest_digest, restored
        destination = directory / "checkpoint"
        if checkpoint_manifest_digest is not None:
            return "p3119-checkpoint"
        saved = save_learning_checkpoint_v3(
            inner.lifecycle.checkpoint(),
            destination,
            optimizer=inner.lifecycle.optimizer,
        )
        checkpoint_manifest_digest = _digest(dict(saved.manifest))
        if arm == "resumed":
            fresh = _new_assembly(mode, objects).lifecycle
            inner.lifecycle = fresh.restore_from_checkpoint(destination)
            restored = True
        return "p3119-checkpoint"

    result = run_learning_loop(
        config=LearningLoopConfig(max_steps=6, checkpoint_every_n_steps=3),
        architecture=inner.lifecycle.architecture,
        architecture_config=inner.lifecycle.architecture_config,
        optimizer=inner.lifecycle.optimizer,
        optimizer_config=inner.lifecycle.optimizer_config,
        optimizer_state=inner.lifecycle.optimizer_state,
        learning_state=inner.lifecycle.learning_state,
        parameters=inner.lifecycle.parameters,
        objective=inner.lifecycle.objective_selection,
        batch_source=source,
        step_executor=executor,
        checkpoint=checkpoint,
        hooks=(hook,),
        emit_run_report=True,
        hf_descriptor=inner.lifecycle.hf_descriptor,
    )
    if (
        result.status != "pass"
        or result.report is None
        or checkpoint_manifest_digest is None
    ):
        raise RuntimeError(f"P3.11.9 {mode} {arm} conveyor failed: {hook.failures}")
    if arm == "resumed" and not restored:
        raise RuntimeError("resumed replay did not use caller-bound restore")
    final = inner.lifecycle
    evidence = ReplayArmEvidence(
        arm=arm,
        experiment_identity=initial_identity,
        lifecycle_identity=_identity(final),
        batch_sequence_digest=_digest(
            [batch.to_dict() for batch in tuple(_batch(index) for index in range(6))]
        ),
        checkpoint_boundary=3,
        checkpoint_manifest_digest=checkpoint_manifest_digest,
        restore_used_caller_identity=restored,
        steps=_step_evidence(records, hook),
        final_parameter_digest=mapping_pytree_digest(final.parameters),
        final_architecture_carry_digest=mapping_pytree_digest(final.architecture_carry),
        final_optimizer_array_digest=mapping_pytree_digest(
            final.optimizer_state.arrays
        ),
        final_optimizer_envelope_digest=_digest(
            final.optimizer_state.envelope.to_dict()
        ),
        final_learning_state_digest=_digest(final.learning_state.to_dict()),
        final_hook_digest=_digest([list(item) for item in hook.events]),
        retained_metrics_digest=_digest(
            [metric.to_dict() for metric in result.metrics]
        ),
        final_report_digest=_digest(result.report.to_dict()),
        final_runtime=_runtime_evidence(result.final_execution),
        non_claims=NON_CLAIMS,
    )
    return _ExecutedArm(
        evidence=evidence,
        parameters=final.parameters,
        architecture_carry=final.architecture_carry,
        optimizer_arrays=final.optimizer_state.arrays,
        learning_state=final.learning_state.to_dict(),
        optimizer_envelope=final.optimizer_state.envelope.to_dict(),
        hook_events=tuple(hook.events),
        retained_metrics=tuple(
            (metric.name, metric.value) for metric in result.metrics
        ),
        logical_paths=tuple(
            (
                tuple(execution.result.changed_parameter_paths),
                tuple(execution.result.unchanged_parameter_paths),
            )
            for _, execution in records
        ),
        rng_coordinates=tuple(step.rng for step in evidence.steps),
        runtimes=tuple(step.runtime for step in evidence.steps),
        final_runtime=evidence.final_runtime,
        report=result.report,
        loop_result=result,
    )


def execute_replay(
    mode: str, root: Path
) -> tuple[ReplayRunEvidence, _ExecutedArm, set[int], list[object]]:
    objects: list[object] = []
    uninterrupted = _run_arm(mode, "uninterrupted", root / "uninterrupted", objects)
    resumed = _run_arm(mode, "resumed", root / "resumed", objects)
    identities = {id(item) for item in objects}
    if len(identities) != len(objects):
        raise RuntimeError("P3.11.9 replay reused a mutable execution object")
    return (
        ReplayRunEvidence(uninterrupted.evidence, resumed.evidence),
        uninterrupted,
        identities,
        objects,
    )


def _mapping_leaves(
    value: Any, prefix: tuple[str, ...] = ()
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    if not isinstance(value, Mapping):
        return ((prefix, value),)
    result: list[tuple[tuple[str, ...], Any]] = []
    for key in sorted(value):
        if not isinstance(key, str) or not key:
            raise RuntimeError("P3.11.9 replay requires string mapping keypaths")
        result.extend(_mapping_leaves(value[key], (*prefix, key)))
    return tuple(result)


def _tree_comparison(
    left: Any, right: Any, *, rtol: float, atol: float
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Compare structure, paths, shapes/dtypes, integers, and floats from execution."""

    left_leaves = _mapping_leaves(left)
    right_leaves = _mapping_leaves(right)
    left_paths = tuple(path for path, _ in left_leaves)
    right_paths = tuple(path for path, _ in right_leaves)
    structure_equal = left_paths == right_paths
    keypaths_equal = left_paths == right_paths
    leaf_count_equal = len(left_leaves) == len(right_leaves)
    dtype_shape_equal = structure_equal
    integer_values_equal = structure_equal
    floating_values_within_tolerance = structure_equal
    if not structure_equal:
        return (
            structure_equal,
            keypaths_equal,
            leaf_count_equal,
            dtype_shape_equal,
            integer_values_equal,
            floating_values_within_tolerance,
        )
    for (_, left_leaf), (_, right_leaf) in zip(left_leaves, right_leaves, strict=True):
        if (
            tuple(left_leaf.shape) != tuple(right_leaf.shape)
            or left_leaf.dtype != right_leaf.dtype
        ):
            dtype_shape_equal = False
            integer_values_equal = False
            floating_values_within_tolerance = False
            continue
        if jnp.issubdtype(left_leaf.dtype, jnp.integer) or jnp.issubdtype(
            left_leaf.dtype, jnp.bool_
        ):
            integer_values_equal = integer_values_equal and bool(
                jnp.all(left_leaf == right_leaf)
            )
        elif jnp.issubdtype(left_leaf.dtype, jnp.floating):
            finite = bool(jnp.all(jnp.isfinite(left_leaf))) and bool(
                jnp.all(jnp.isfinite(right_leaf))
            )
            floating_values_within_tolerance = (
                floating_values_within_tolerance
                and finite
                and bool(jnp.allclose(left_leaf, right_leaf, rtol=rtol, atol=atol))
            )
        else:
            integer_values_equal = False
            floating_values_within_tolerance = False
    return (
        structure_equal,
        keypaths_equal,
        leaf_count_equal,
        dtype_shape_equal,
        integer_values_equal,
        floating_values_within_tolerance,
    )


def _metrics_comparison(
    left: tuple[tuple[str, Any], ...],
    right: tuple[tuple[str, Any], ...],
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, bool]:
    left_names = tuple(name for name, _ in left)
    right_names = tuple(name for name, _ in right)
    if left_names != right_names:
        return False, False
    for (_, left_value), (_, right_value) in zip(left, right, strict=True):
        if isinstance(left_value, int) and not isinstance(left_value, bool):
            if type(left_value) is not type(right_value) or left_value != right_value:
                return True, False
            continue
        try:
            left_float = float(left_value)
            right_float = float(right_value)
        except (TypeError, ValueError):
            return True, False
        if not (
            bool(jnp.isfinite(left_float)) and bool(jnp.isfinite(right_float))
        ) or not bool(jnp.allclose(left_float, right_float, rtol=rtol, atol=atol)):
            return True, False
    return True, True


def _runtime_structure_equal(left: RuntimeEvidence, right: RuntimeEvidence) -> bool:
    if (
        left.mode != "eager"
        or right.mode != "jit"
        or left.compiled
        or not right.compiled
    ):
        return False
    eager = left.to_dict()
    jit = right.to_dict()
    eager.pop("mode")
    eager.pop("compiled")
    jit.pop("mode")
    jit.pop("compiled")
    return eager == jit


def _compare_executed_cross_mode(
    eager: _ExecutedArm, jit: _ExecutedArm, tolerance: ToleranceEvidence
) -> CrossModeComparisonEvidence:
    rtol = parse_finite_float_hex(tolerance.rtol, positive=True)
    atol = parse_finite_float_hex(tolerance.atol, positive=True)
    comparisons = (
        _tree_comparison(eager.parameters, jit.parameters, rtol=rtol, atol=atol),
        _tree_comparison(
            eager.architecture_carry, jit.architecture_carry, rtol=rtol, atol=atol
        ),
        _tree_comparison(
            eager.optimizer_arrays, jit.optimizer_arrays, rtol=rtol, atol=atol
        ),
    )
    metric_names_equal, metric_values_within_tolerance = _metrics_comparison(
        eager.retained_metrics, jit.retained_metrics, rtol=rtol, atol=atol
    )
    return CrossModeComparisonEvidence(
        structure_equal=all(item[0] for item in comparisons),
        keypaths_equal=all(item[1] for item in comparisons),
        leaf_count_equal=all(item[2] for item in comparisons),
        dtype_shape_equal=all(item[3] for item in comparisons),
        integer_values_equal=all(item[4] for item in comparisons),
        floating_values_within_tolerance=all(item[5] for item in comparisons),
        learning_state_equal=eager.learning_state == jit.learning_state,
        optimizer_envelope_equal=eager.optimizer_envelope == jit.optimizer_envelope,
        lifecycle_identity_equal=(
            eager.evidence.lifecycle_identity == jit.evidence.lifecycle_identity
        ),
        hook_sequence_equal=eager.hook_events == jit.hook_events,
        metric_names_equal=metric_names_equal,
        metric_values_within_tolerance=metric_values_within_tolerance,
        logical_paths_equal=eager.logical_paths == jit.logical_paths,
        rng_identity_equal=eager.rng_coordinates == jit.rng_coordinates,
        runtime_structure_equal=(
            len(eager.runtimes) == len(jit.runtimes)
            and all(
                _runtime_structure_equal(left, right)
                for left, right in zip(eager.runtimes, jit.runtimes, strict=True)
            )
            and _runtime_structure_equal(eager.final_runtime, jit.final_runtime)
        ),
        declared_rtol=tolerance.rtol,
        declared_atol=tolerance.atol,
    )


def execute_stateful_replays(root: Path) -> StatefulReplayProof:
    all_objects: list[object] = []
    modes: dict[str, dict[str, ReplayRunEvidence]] = {}
    captured: dict[str, dict[str, _ExecutedArm]] = {}
    replay_ids: list[set[int]] = []
    for mode in ("eager", "jit"):
        runs: dict[str, ReplayRunEvidence] = {}
        mode_captured: dict[str, _ExecutedArm] = {}
        for label in ("replay_a", "replay_b"):
            run, uninterrupted, identities, objects = execute_replay(
                mode, root / mode / label
            )
            runs[label] = run
            mode_captured[label] = uninterrupted
            replay_ids.append(identities)
            all_objects.extend(objects)
        modes[mode] = runs
        captured[mode] = mode_captured
    for index, current in enumerate(replay_ids):
        for prior in replay_ids[:index]:
            if current & prior:
                raise RuntimeError("P3.11.9 independent replays shared mutable objects")
    eager = modes["eager"]["replay_a"].uninterrupted
    tolerance = ToleranceEvidence(
        rtol=finite_float_hex(1e-6), atol=finite_float_hex(1e-6)
    )
    cross_mode = _compare_executed_cross_mode(
        captured["eager"]["replay_a"], captured["jit"]["replay_a"], tolerance
    )
    return StatefulReplayProof(
        experiment_identity=eager.experiment_identity,
        replay_count=2,
        tolerance=tolerance,
        modes=modes,
        cross_mode=cross_mode,
        non_claims=NON_CLAIMS,
        executed_cross_mode=cross_mode,
    )


__all__ = ["NON_CLAIMS", "StatefulLinearJaxArchitecture", "execute_stateful_replays"]
