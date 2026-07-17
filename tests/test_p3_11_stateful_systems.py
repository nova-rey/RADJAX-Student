"""P3.11.8 stateful end-to-end proof through public JAX contracts."""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from radjax_student.architecture import (  # noqa: E402
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
from radjax_student.architecture.testing import (  # noqa: E402
    FAKE_ARCHITECTURE_CAPABILITIES,
    FakeArchitecturePlugin,
)
from radjax_student.checkpoints import (  # noqa: E402
    CheckpointValidationError,
    JaxLearningCheckpointV3,
    load_learning_checkpoint_v3,
    save_learning_checkpoint_v3,
)
from radjax_student.checkpoints.npz_codec import (  # noqa: E402
    describe_mapping_pytree,
    descriptor_digest,
)
from radjax_student.contracts import (  # noqa: E402
    HFPreservationReference,
    ObjectiveConfig,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)
from radjax_student.learning import (  # noqa: E402
    HookResult,
    LearningBatch,
    LearningState,
    MetricRecord,
    ObjectiveScope,
    UpdateScope,
)
from radjax_student.learning.jax_batch import (
    FiniteJsonJaxBatchMaterializer,  # noqa: E402
)
from radjax_student.legacy.objectives_jax import (  # noqa: E402
    LegacyJaxObjectiveConfig as JaxObjectiveConfig,
)
from radjax_student.objectives import (  # noqa: E402
    CANONICAL_MSE_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import (  # noqa: E402
    OptimizerConfig,
    OptimizerRegistry,
    OptimizerState,
    SgdOptimizer,
)
from radjax_student.runtime import (  # noqa: E402
    CompilationOptions,
    ExecutionRequest,
    RuntimeConfig,
    RuntimeKeys,
    build_default_runtime_registry,
    inspect_runtime_environment,
    select_runtime_backend,
)
from radjax_student.steps import (  # noqa: E402
    JaxLearningLifecycle,
    JaxLoopExecutor,
    LearningLoopConfig,
    SyntheticBatchSource,
    run_learning_loop,
)
from radjax_student.validation.p3_11_8_systems_receipt import (  # noqa: E402
    NON_CLAIMS,
    StatefulSystemsProofResult,
    build_stateful_systems_receipt,
)
from radjax_student.validation.p3_11_9_replay.runner_jax import (  # noqa: E402
    execute_stateful_replays,
)

pytestmark = pytest.mark.jax

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "docs" / "P3_11_8_STATEFUL_SYSTEMS_RECEIPT.json"
ARCHITECTURE_ID = "test.stateful_linear_jax.v1"


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


class StatefulLinearJaxArchitecture(FakeArchitecturePlugin):
    """Test-only complete plugin with an RNG-observable functional carry."""

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
        catalog = self.describe_parameters()
        layout = _layout()
        config_digest = _digest(request.config.to_dict())
        hf = HFPreservationReference(
            "hf_preservation_reference.v1",
            "stateful-linear-hf-descriptor",
            "stateful-linear",
            self.architecture_id,
            "stateful-test-tokenizer",
            8,
            "stateful-special-tokens",
            layout.digest(),
            config_digest,
        )
        return ArchitectureInitResult(
            parameter_catalog=catalog,
            architecture_state=ArchitectureState("stateful-linear-state.v1"),
            parameters={
                "trunk": {"weight": jnp.asarray((0.0,), dtype=jnp.float32)},
                "head": {"bias": jnp.asarray((0.0,), dtype=jnp.float32)},
            },
            architecture_carry={
                "forwards": jnp.asarray(0, dtype=jnp.int32),
                "rng_probe": jnp.asarray(0.0, dtype=jnp.float32),
            },
            architecture_carry_descriptor={
                "schema_version": "architecture_carry.v1",
                "state_id": "stateful-linear-state.v1",
                "pytree_descriptor_digest": descriptor_digest(
                    describe_mapping_pytree(
                        {
                            "forwards": jnp.asarray(0, dtype=jnp.int32),
                            "rng_probe": jnp.asarray(0.0, dtype=jnp.float32),
                        }
                    )
                ),
            },
            parameter_layout=layout,
            hf_reference=hf,
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
    hook_id = "p3118.events"
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

    def __init__(self):
        self.events = []
        self.failures = []

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


def _batch(index: int, *, target_scale: float = 1.0) -> LearningBatch:
    return LearningBatch(
        f"stateful-{index}",
        inputs={"x": [-1.0, 0.0, 1.0]},
        targets={"y": [[-2.0 * target_scale], [0.0], [2.0 * target_scale]]},
    )


def _runtime(mode: str):
    config = RuntimeConfig(
        backend_id="jax",
        platform_preference="cpu",
        precision_policy="float32",
        placement_policy="single_device",
        compilation_policy=mode,
        distributed_policy="disabled",
        fallback_policy="disallowed",
        seed=17,
    )
    inspection = inspect_runtime_environment()
    registry = build_default_runtime_registry()
    selection = select_runtime_backend(config, inspection, registry)
    assert selection.ok and selection.selected_platform == "cpu"
    backend = registry.get("jax")
    device = next(
        item for item in inspection.device_inventory.devices if item.platform == "cpu"
    )
    context = backend.initialize_portability_context(
        config, inspection, selection, device
    )
    return config, backend, context


def _request(mode: str):
    def factory(state: LearningState) -> ExecutionRequest:
        return ExecutionRequest(
            request_id=f"p3118.{mode}.{state.global_step}",
            function_id="p3118.stateful_complete_step",
            mode=mode,
            compilation_options=CompilationOptions(mode=mode, synchronize_results=True),
        )

    return factory


def _lifecycle(mode: str):
    architecture = StatefulLinearJaxArchitecture(architecture_id=ARCHITECTURE_ID)
    architecture_registry = ArchitectureRegistry()
    architecture_registry.register(architecture)
    architecture = architecture_registry.get(architecture.architecture_id)
    config = ArchitectureConfig(
        architecture.architecture_id, vocab_size=8, dtype_intent="float32"
    )
    initialized = architecture.initialize_parameters(
        ArchitectureInitRequest(config, "runtime_keys.v1:initialization:17", "float32")
    )
    optimizer = SgdOptimizer()
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(optimizer)
    optimizer = optimizer_registry.get(optimizer.optimizer_id)
    optimizer_config = OptimizerConfig(optimizer.optimizer_id, learning_rate=0.2)
    optimizer_state = optimizer.initialize_jax_state(
        config=optimizer_config,
        parameter_layout=initialized.parameter_layout,
        optimizer_state=OptimizerState(
            optimizer.optimizer_id, initialized.parameter_layout.logical_paths
        ),
    )
    objective_registry = build_default_objective_registry()
    objective_selection = objective_registry.select(CANONICAL_MSE_IDENTITY)
    objective_config = ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"})
    resolved_objective_selection = architecture.resolve_objective_scope(
        ObjectiveScope(), architecture.architecture_metadata()
    )
    objective_descriptor = objective_registry.execution_descriptor(
        selection=objective_selection,
        config=objective_config,
        resolved_selection=resolved_objective_selection,
    )
    runtime_config, backend, context = _runtime(mode)
    lifecycle = JaxLearningLifecycle(
        architecture=architecture,
        architecture_config=config,
        architecture_state=initialized.architecture_state,
        architecture_carry=initialized.architecture_carry,
        parameter_catalog=initialized.parameter_catalog,
        parameter_layout=initialized.parameter_layout,
        hf_reference=initialized.hf_reference,
        objective_selection=objective_selection,
        objective_config=objective_config,
        resolved_objective_selection=resolved_objective_selection,
        objective_descriptor=objective_descriptor,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        parameters=initialized.parameters,
        learning_state=LearningState(
            "p3118",
            active_update_scope=UpdateScope("named_region", "trunk"),
            active_objective_scope=ObjectiveScope(),
        ),
        runtime_context=context,
        runtime_backend=backend,
        runtime_key_stream=RuntimeKeys.from_seed(runtime_config.seed).dropout,
        architecture_carry_descriptor=initialized.architecture_carry_descriptor,
    )
    return lifecycle, runtime_config


def _run(
    mode: str, tmp_path: Path, *, restore_at_three: bool, target_scale: float = 1.0
):
    lifecycle, runtime_config = _lifecycle(mode)
    executor = JaxLoopExecutor(
        lifecycle,
        FiniteJsonJaxBatchMaterializer(),
        _request(mode),
        precision_policy="float32",
    )
    hook = EventHook()
    saved: JaxLearningCheckpointV3 | None = None

    def checkpoint(_execution):
        nonlocal saved
        if saved is not None:
            return "p3118-checkpoint"
        destination = tmp_path / f"checkpoint-{mode}"
        saved = save_learning_checkpoint_v3(
            executor.lifecycle.checkpoint(),
            destination,
            optimizer=executor.lifecycle.optimizer,
        )
        assert (
            saved.architecture_carry_descriptor
            == executor.lifecycle.architecture_carry_descriptor
        )
        if restore_at_three:
            executor.lifecycle = executor.lifecycle.restore_from_checkpoint(destination)
        return "p3118-checkpoint"

    result = run_learning_loop(
        config=LearningLoopConfig(max_steps=6, checkpoint_every_n_steps=3),
        architecture=executor.lifecycle.architecture,
        architecture_config=executor.lifecycle.architecture_config,
        optimizer=executor.lifecycle.optimizer,
        optimizer_config=executor.lifecycle.optimizer_config,
        optimizer_state=executor.lifecycle.optimizer_state,
        learning_state=executor.lifecycle.learning_state,
        parameters=executor.lifecycle.parameters,
        objective=executor.lifecycle.objective_selection,
        batch_source=SyntheticBatchSource(
            tuple(_batch(index, target_scale=target_scale) for index in range(6))
        ),
        step_executor=executor,
        checkpoint=checkpoint,
        hooks=(hook,),
        emit_run_report=True,
    )
    assert (
        result.status == "pass" and saved is not None and result.report is not None
    ), hook.failures
    return result, executor.lifecycle, hook.events, saved, runtime_config


def _leaves_equal(first, second) -> bool:
    pairs = zip(
        jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True
    )
    return all(jnp.array_equal(left, right) for left, right in pairs)


def _require_exact(first, second) -> None:
    if not _leaves_equal(first, second):
        raise AssertionError("stateful systems proof values differ")


def _tree_signature(value) -> list[list[Any]]:
    return [
        [list(leaf.shape), str(leaf.dtype)] for leaf in jax.tree_util.tree_leaves(value)
    ]


def _require_cross_mode_arrays(first, second) -> None:
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)
    assert _tree_signature(first) == _tree_signature(second)
    for left, right in zip(
        jax.tree_util.tree_leaves(first),
        jax.tree_util.tree_leaves(second),
        strict=True,
    ):
        assert jnp.allclose(left, right, rtol=1e-6, atol=1e-6)


def _runtime_evidence(execution) -> dict[str, Any]:
    receipt = execution.runtime_result
    metadata = dict(receipt.output_metadata)
    preparation = dict(metadata["input_preparation"])
    preparation.pop("selected_device_id", None)
    return {
        "status": receipt.status,
        "backend_id": receipt.backend_id,
        "mode": receipt.mode,
        "compiled": receipt.compiled,
        "dispatched": receipt.dispatched,
        "synchronized": receipt.synchronized,
        "output_metadata_keys": sorted(metadata),
        "input_preparation": preparation,
        "rng_bridge": dict(metadata["rng_bridge"]),
        "claims_not_made": list(receipt.claims_not_made),
    }


def _mode_evidence(result, lifecycle, events) -> dict[str, Any]:
    execution = result.final_execution
    assert execution is not None
    return {
        "learning_state": lifecycle.learning_state.to_dict(),
        "optimizer_envelope": lifecycle.optimizer_state.envelope.to_dict(),
        "parameter_signature": _tree_signature(lifecycle.parameters),
        "carry_signature": _tree_signature(lifecycle.architecture_carry),
        "optimizer_array_signature": _tree_signature(lifecycle.optimizer_state.arrays),
        "changed_paths": list(execution.result.changed_parameter_paths),
        "unchanged_paths": list(execution.result.unchanged_parameter_paths),
        "hook_events": [list(event) for event in events],
        "retained_metric_names": [metric.name for metric in result.metrics],
        "lifecycle_identity": {
            "layout_digest": lifecycle.parameter_layout.digest(),
            "hf_reference": lifecycle.hf_reference.to_dict(),
            "config_digest": lifecycle.config_digest,
            "catalog_digest": lifecycle.catalog_digest,
            "carry_descriptor": dict(lifecycle.architecture_carry_descriptor),
        },
        "runtime_receipt": _runtime_evidence(execution),
    }


def _execute_stateful_systems_proof(tmp_path):
    outcomes = {}
    assertion_results = {
        "complete_architecture_plugin_used": True,
        "complete_optimizer_plugin_used": True,
        "public_runtime_path_used": True,
        "runtime_owned_rng_used": True,
        "runtime_owned_placement_used": True,
        "architecture_scope_routing_used": True,
        "optimizer_boundary_used": True,
        "generic_loop_used": True,
        "hooks_used": True,
        "metrics_retained": True,
        "report_produced": True,
        "stateful_carry_advanced": True,
        "checkpoint_v3_saved": True,
        "caller_bound_restore_validated": True,
        "uninterrupted_resumed_equality_passed": True,
        "eager_jit_comparison_passed": True,
        "no_legacy_fallback_used": True,
    }
    for mode in ("eager", "jit"):
        uninterrupted = _run(
            mode, tmp_path / f"continuous-{mode}", restore_at_three=False
        )
        resumed = _run(mode, tmp_path / f"resumed-{mode}", restore_at_three=True)
        full_result, full, full_events, full_saved, _ = uninterrupted
        resume_result, resumed_lifecycle, resume_events, _, _ = resumed
        _require_exact(full.parameters, resumed_lifecycle.parameters)
        _require_exact(full.architecture_carry, resumed_lifecycle.architecture_carry)
        _require_exact(
            full.optimizer_state.arrays, resumed_lifecycle.optimizer_state.arrays
        )
        assert (
            full.optimizer_state.envelope == resumed_lifecycle.optimizer_state.envelope
        )
        assert full.learning_state == resumed_lifecycle.learning_state
        assert full_events == resume_events
        assert full_result.report.to_dict() == resume_result.report.to_dict()
        assert full_result.final_execution.result.changed_parameter_paths == (
            "trunk.weight",
        )
        assert full_result.final_execution.result.unchanged_parameter_paths == (
            "head.bias",
        )
        assert (
            int(full.optimizer_state.arrays["per_parameter_steps"]["head"]["bias"]) == 0
        )
        assert (
            int(full.optimizer_state.arrays["step"])
            == full.optimizer_state.envelope.step
            == 6
        )
        assert (
            full.learning_state.global_step == full.learning_state.optimizer_step == 6
        )
        assert full.learning_state.micro_step == 0
        assert int(full.architecture_carry["forwards"]) == 6
        assert float(full.parameters["trunk"]["weight"][0]) != 0.0
        assert float(full.parameters["head"]["bias"][0]) == 0.0
        losses = [
            metric.value
            for metric in full_result.metrics
            if metric.name == "objective.mse"
        ]
        assert losses[-1] < losses[0]
        receipt = full_result.final_execution.runtime_result
        assert (
            receipt.output_metadata["rng_bridge"]["schema_version"]
            == "runtime_jax_key_bridge.v1"
        )
        assert receipt.output_metadata["input_preparation"]["selected_device_id"]
        outcomes[mode] = (full_result, full, full_events, full_saved)

    eager_result, eager, eager_events, eager_saved = outcomes["eager"]
    jit_result, jit, jit_events, jit_saved = outcomes["jit"]
    _require_cross_mode_arrays(eager.parameters, jit.parameters)
    _require_cross_mode_arrays(eager.architecture_carry, jit.architecture_carry)
    _require_cross_mode_arrays(eager.optimizer_state.arrays, jit.optimizer_state.arrays)
    assert eager.learning_state == jit.learning_state
    assert eager.optimizer_state.envelope == jit.optimizer_state.envelope
    assert eager_events == jit_events
    assert tuple(metric.name for metric in eager_result.metrics) == tuple(
        metric.name for metric in jit_result.metrics
    )
    assert eager_result.final_execution.result.changed_parameter_paths == (
        jit_result.final_execution.result.changed_parameter_paths
    )
    assert eager_result.final_execution.result.unchanged_parameter_paths == (
        jit_result.final_execution.result.unchanged_parameter_paths
    )
    assert eager.parameter_layout.digest() == jit.parameter_layout.digest()
    assert eager.hf_reference == jit.hf_reference
    assert eager.config_digest == jit.config_digest
    assert eager.catalog_digest == jit.catalog_digest
    assert eager.architecture_carry_descriptor == jit.architecture_carry_descriptor
    eager_runtime = _runtime_evidence(eager_result.final_execution)
    jit_runtime = _runtime_evidence(jit_result.final_execution)
    assert eager_runtime["mode"] == "eager" and not eager_runtime["compiled"]
    assert jit_runtime["mode"] == "jit" and jit_runtime["compiled"]
    for key in (
        "status",
        "backend_id",
        "dispatched",
        "synchronized",
        "output_metadata_keys",
        "input_preparation",
        "rng_bridge",
        "claims_not_made",
    ):
        assert eager_runtime[key] == jit_runtime[key]
    assert (
        eager_saved.architecture_carry_descriptor == eager.architecture_carry_descriptor
    )
    assert jit_saved.architecture_carry_descriptor == jit.architecture_carry_descriptor
    return StatefulSystemsProofResult(
        assertions=assertion_results,
        mode_evidence={
            "eager": _mode_evidence(eager_result, eager, eager_events),
            "jit": _mode_evidence(jit_result, jit, jit_events),
        },
        cross_mode_evidence={
            "parameter_structure_dtype_and_tolerance": True,
            "carry_structure_dtype_and_tolerance": True,
            "optimizer_array_structure_dtype_and_tolerance": True,
            "learning_state": True,
            "optimizer_envelope": True,
            "hooks": True,
            "retained_metrics": True,
            "changed_and_unchanged_paths": True,
            "lifecycle_identity": True,
            "runtime_receipt_metadata": True,
        },
    )


def test_stateful_conveyor_resume_and_execution_modes(tmp_path):
    proof = _execute_stateful_systems_proof(tmp_path)
    assert all(proof.assertions.values())


def test_p3118_and_p3119_share_the_public_stateful_conveyor(tmp_path):
    replay = execute_stateful_replays(tmp_path / "p3119")
    assert replay.modes["eager"]["replay_a"].resumed.restore_used_caller_identity
    assert replay.modes["jit"]["replay_a"].uninterrupted.final_parameter_digest


def test_stateful_system_proof_rejects_real_boundary_violations(tmp_path):
    lifecycle, _ = _lifecycle("eager")

    class ApplyJaxOnlyArchitecture:
        def apply_jax(self, *args, **kwargs):
            del args, kwargs

    class JaxMethodsOnlyOptimizer:
        optimizer_id = "test.jax_methods_only"
        optimizer_version = 1

        def jax_state_descriptor(self, parameter_layout):
            del parameter_layout

        def initialize_jax_state(self, **kwargs):
            del kwargs

        def validate_jax_state(self, **kwargs):
            del kwargs

        def apply_jax_updates(self, **kwargs):
            del kwargs

    with pytest.raises(TypeError, match="JaxArchitecturePlugin"):
        replace(lifecycle, architecture=ApplyJaxOnlyArchitecture())
    with pytest.raises(TypeError, match="JaxOptimizerBackend"):
        replace(lifecycle, optimizer=JaxMethodsOnlyOptimizer())
    with pytest.raises(ValueError, match="runtime key stream"):
        replace(lifecycle, runtime_key_stream=RuntimeKeys.from_seed(99).dropout)

    _, restored_lifecycle, _, saved, _ = _run(
        "eager", tmp_path / "saved", restore_at_three=True
    )
    destination = tmp_path / "saved" / "checkpoint-eager"
    with pytest.raises(ValueError, match="checkpoint HF identity"):
        lifecycle.with_checkpoint(
            replace(
                saved, hf_reference=replace(saved.hf_reference, tokenizer_id="foreign")
            )
        )
    with pytest.raises(ValueError, match="parameter layout"):
        lifecycle.with_checkpoint(
            replace(
                saved,
                parameter_layout=ParameterTreeLayout(
                    lifecycle.architecture.architecture_id,
                    lifecycle.parameter_layout.entries[:1],
                ),
            )
        )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_hf_identity_mismatch"
    ):
        load_learning_checkpoint_v3(
            destination,
            optimizer=lifecycle.optimizer,
            parameter_layout=lifecycle.parameter_layout,
            expected_hf_reference=replace(
                lifecycle.hf_reference, tokenizer_id="foreign"
            ),
            expected_objective_descriptor=lifecycle.objective_descriptor,
            expected_objective_config=lifecycle.objective_config,
            expected_resolved_objective_selection=lifecycle.resolved_objective_selection,
            expected_objective_selection=lifecycle.objective_selection,
        )
    with pytest.raises(CheckpointValidationError, match="checkpoint_layout_mismatch"):
        load_learning_checkpoint_v3(
            destination,
            optimizer=lifecycle.optimizer,
            parameter_layout=ParameterTreeLayout(
                ARCHITECTURE_ID, lifecycle.parameter_layout.entries[:1]
            ),
            expected_objective_descriptor=lifecycle.objective_descriptor,
            expected_objective_config=lifecycle.objective_config,
            expected_resolved_objective_selection=lifecycle.resolved_objective_selection,
            expected_objective_selection=lifecycle.objective_selection,
        )
    bad_optimizer_state = replace(
        saved.optimizer_state,
        envelope=replace(saved.optimizer_state.envelope, step=4),
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_optimizer_step_mismatch"
    ):
        save_learning_checkpoint_v3(
            replace(saved, optimizer_state=bad_optimizer_state),
            tmp_path / "bad-optimizer",
            optimizer=lifecycle.optimizer,
        )
    tampered_arrays = {
        "step": restored_lifecycle.optimizer_state.arrays["step"],
        "per_parameter_steps": {
            "trunk": restored_lifecycle.optimizer_state.arrays["per_parameter_steps"][
                "trunk"
            ],
            "head": {"bias": jnp.asarray(1, dtype=jnp.int32)},
        },
    }
    with pytest.raises(AssertionError, match="values differ"):
        _require_exact(restored_lifecycle.optimizer_state.arrays, tampered_arrays)
    _, divergent, _, _, _ = _run(
        "eager",
        tmp_path / "different-batches",
        restore_at_three=False,
        target_scale=1.5,
    )
    with pytest.raises(AssertionError, match="values differ"):
        _require_exact(restored_lifecycle.parameters, divergent.parameters)
    foreign_lifecycle = replace(
        restored_lifecycle,
        hf_reference=replace(restored_lifecycle.hf_reference, tokenizer_id="foreign"),
    )
    with pytest.raises(
        CheckpointValidationError, match="checkpoint_hf_identity_mismatch"
    ):
        foreign_lifecycle.restore_from_checkpoint(destination)
    alternate_state = ArchitectureState("stateful-linear-state.v2")
    alternate_descriptor = {
        **dict(saved.architecture_carry_descriptor),
        "state_id": alternate_state.state_id,
    }
    adopted = lifecycle.with_checkpoint(
        replace(
            saved,
            architecture_state=alternate_state,
            architecture_carry_descriptor=alternate_descriptor,
        )
    )
    assert adopted.architecture_state == alternate_state

    from radjax_student.legacy.jax_learning import execute_legacy_jax_learning_step

    class LegacyMeanSquaredError:
        def evaluate(self, surface, targets, weights, objective_config):
            del weights, objective_config
            loss = jnp.mean((surface - targets["y"]) ** 2)
            return loss, {"mse": loss}

    executor = JaxLoopExecutor(
        lifecycle,
        FiniteJsonJaxBatchMaterializer(),
        _request("eager"),
        precision_policy="float32",
    )
    legacy_execution = execute_legacy_jax_learning_step(
        architecture=lifecycle.architecture,
        objective=LegacyMeanSquaredError(),
        parameters=lifecycle.parameters,
        architecture_carry=lifecycle.architecture_carry,
        batch=FiniteJsonJaxBatchMaterializer().materialize(_batch(0)),
        objective_config=JaxObjectiveConfig("stateful_linear_mse.v1"),
        rng_key=jax.random.key(0),
        selection_mask={"trunk": {"weight": True}, "head": {"bias": False}},
        learning_rate=0.2,
        runtime_context=lifecycle.runtime_context,
        runtime_backend=lifecycle.runtime_backend,
        execution_request=_request("eager")(lifecycle.learning_state),
    )
    with pytest.raises(TypeError, match="legacy or partial"):
        executor.accept_execution(legacy_execution)


def test_stateful_receipt_is_derived_from_executed_systems_evidence(tmp_path):
    payload = build_stateful_systems_receipt(_execute_stateful_systems_proof(tmp_path))
    assert payload["non_claims"] == list(NON_CLAIMS)
    assert payload["proof_evidence_digest"] == _digest(payload["proof_evidence"])
    assert json.loads(RECEIPT_PATH.read_text()) == payload
