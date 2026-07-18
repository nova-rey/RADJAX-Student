"""Focused P3.12C assembly tests; this module imports JAX only in test bodies."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from radjax_student.architecture import ArchitectureConfig, ArchitectureRegistry
from radjax_student.contracts import ObjectiveConfig
from radjax_student.learning import (
    JaxLearningAssemblyRegistries,
    JaxLearningAssemblyRequest,
    LearningState,
    ObjectiveScope,
    UpdateScope,
    assemble_jax_learning_lifecycle,
)
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    build_default_objective_registry,
)
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry, SgdOptimizer
from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry
from radjax_student.validation.p3_11_9_replay.runner_jax import (
    ARCHITECTURE_ID,
    StatefulLinearJaxArchitecture,
    _batch,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly import (
    experiments,
    implementation_audit,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly.diagnostic import (
    observe,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly.inventory import (
    ADVERSARIAL_CASE_IDS,
    POSITIVE_CASE_IDS,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly.models import (
    build_receipt,
    validate_receipt,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly.runner_jax import (
    execute_lifecycle_assembly_proof,
    execute_raw_diagnostic,
)

pytestmark = pytest.mark.jax


def _receipt_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _request_and_registries():
    architecture_registry = ArchitectureRegistry()
    architecture_registry.register(StatefulLinearJaxArchitecture(ARCHITECTURE_ID))
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(SgdOptimizer())
    request = JaxLearningAssemblyRequest(
        architecture_id=ARCHITECTURE_ID,
        architecture_version=1,
        architecture_config=ArchitectureConfig(
            ARCHITECTURE_ID, vocab_size=8, dtype_intent="float32"
        ),
        objective_identity=CANONICAL_MSE_IDENTITY,
        objective_config=ObjectiveConfig(CANONICAL_MSE_IDENTITY, {"reduction": "mean"}),
        optimizer_id="sgd.v1",
        optimizer_version=1,
        optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.25),
        runtime_backend_id="jax",
        runtime_implementation_version="p2.9",
        runtime_config=RuntimeConfig(
            backend_id="jax",
            platform_preference="cpu",
            precision_policy="float32",
            placement_policy="single_device",
            compilation_policy="eager",
            distributed_policy="disabled",
            fallback_policy="disallowed",
            seed=17,
        ),
        root_seed=17,
        learning_state=LearningState(
            "p312c",
            active_update_scope=UpdateScope("named_region", "trunk"),
            active_objective_scope=ObjectiveScope(),
        ),
    )
    return request, JaxLearningAssemblyRegistries(
        architecture_registry,
        build_default_objective_registry(),
        optimizer_registry,
        build_default_runtime_registry(),
    )


def test_inventory_is_exact_and_ordered():
    assert len(POSITIVE_CASE_IDS) == 17
    assert len(ADVERSARIAL_CASE_IDS) == 36
    assert POSITIVE_CASE_IDS[-1] == "repeated_assembly_identity"


def test_request_round_trip_rejects_unknown_fields():
    request, _ = _request_and_registries()
    assert JaxLearningAssemblyRequest.from_dict(request.to_dict()) == request
    payload = request.to_dict() | {"passed": True}
    with pytest.raises(Exception, match="unknown"):
        JaxLearningAssemblyRequest.from_dict(payload)


def test_request_cannot_bypass_frozen_invariants_at_assembly_boundary():
    request, registries = _request_and_registries()
    object.__setattr__(request, "architecture_id", "")
    with pytest.raises(Exception, match="assembly_architecture_invalid"):
        assemble_jax_learning_lifecycle(request, registries=registries)


def test_assembler_is_deterministic_and_audit_passes():
    request, registries = _request_and_registries()
    first = assemble_jax_learning_lifecycle(request, registries=registries)
    second = assemble_jax_learning_lifecycle(request, registries=registries)
    assert first.assembly_digest == second.assembly_digest
    assert first.loop_executor.lifecycle is first.lifecycle
    assert implementation_audit.audit_assembly_authority(Path.cwd()).status == "pass"


def test_result_summary_is_derived_from_selected_lifecycle_components():
    request, registries = _request_and_registries()
    result = assemble_jax_learning_lifecycle(request, registries=registries)
    with pytest.raises(Exception, match="assembly_identity_mismatch"):
        replace(result, summary={**result.summary, "architecture_identity": "other"})


def test_assembled_executor_runs_one_real_jax_step():
    request, registries = _request_and_registries()
    assembled = assemble_jax_learning_lifecycle(request, registries=registries)
    before = assembled.lifecycle
    execution = assembled.loop_executor(
        architecture=before.architecture,
        architecture_config=before.architecture_config,
        optimizer=before.optimizer,
        optimizer_config=before.optimizer_config,
        optimizer_state=before.optimizer_state,
        learning_state=before.learning_state,
        parameters=before.parameters,
        objective=before.objective_selection,
        batch=_batch(0),
    )
    after = assembled.loop_executor.lifecycle
    assert execution.result.loss.loss >= 0.0
    assert after.learning_state.global_step == before.learning_state.global_step + 1
    assert (
        after.learning_state.optimizer_step == before.learning_state.optimizer_step + 1
    )
    assert (
        after.optimizer_state.envelope.step == before.optimizer_state.envelope.step + 1
    )


def test_initial_literal_adversaries_observe_real_assembly_errors():
    cases = (
        (experiments.experiment_wrong_request_type, "assembly_request_invalid"),
        (
            experiments.experiment_empty_architecture_identity,
            "assembly_architecture_invalid",
        ),
        (
            experiments.experiment_unknown_architecture_identity,
            "assembly_architecture_unknown",
        ),
        (
            experiments.experiment_architecture_initialization_incomplete,
            "assembly_architecture_result_invalid",
        ),
        (
            experiments.experiment_unknown_optimizer_identity,
            "assembly_optimizer_unknown",
        ),
    )
    for experiment, expected_code in cases:
        first = observe(experiment())
        second = observe(experiment())
        assert first.boundary == (
            "radjax_student.learning.assembly.assemble_jax_learning_lifecycle"
        )
        assert first.code == expected_code
        assert second.code == expected_code
        assert first.evidence_digest == second.evidence_digest


def test_exact_36_adversarial_diagnostic_is_fresh_and_exact():
    results = execute_raw_diagnostic()
    assert tuple(item.case_id for item in results) == ADVERSARIAL_CASE_IDS
    assert all(item.mutation_applied for item in results)
    assert all(item.deterministic_first_failure for item in results)
    assert all(item.observed_boundary == item.intended_boundary for item in results)
    assert all(item.observed_code == item.expected_code for item in results)
    assert all(item.outcome == "reject" for item in results)


def test_full_production_proof_executes_step_checkpoint_report_and_receipt(tmp_path):
    proof = execute_lifecycle_assembly_proof(tmp_path)
    receipt = build_receipt(proof)
    assert validate_receipt(receipt) == receipt
    assert receipt["positive_proof_count"] == 17
    assert receipt["adversarial_case_count"] == 36


def test_receipt_generation_is_byte_identical_from_fresh_directories(tmp_path):
    first = build_receipt(execute_lifecycle_assembly_proof(tmp_path / "first"))
    second = build_receipt(execute_lifecycle_assembly_proof(tmp_path / "second"))
    assert _receipt_bytes(first) == _receipt_bytes(second)
