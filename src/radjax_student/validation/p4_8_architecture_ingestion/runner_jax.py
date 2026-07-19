"""Lazy-JAX P4.8 end-to-end architecture-ingestion evidence generator."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from radjax_student.validation.architecture_audit import (
    build_phase4_architecture_ingestion_audit,
)
from radjax_student.validation.p4_8_architecture_ingestion.models import (
    EQUATION_PARITY_CLAIM,
    NON_CLAIM,
    REQUIRED_NON_CLAIMS,
    SCHEMA_VERSION,
    canonical_report_bytes,
    derive_status,
)

_TOKENS = (1, 7, 3, 5)
_TARGETS = (7, 3, 5, 1)
_NEXT_TOKENS = (5, 3, 7, 1)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _assembly(compilation_policy: str) -> Any:
    """Build the one real explicit registry-selected RWKV lifecycle."""

    from radjax_student.architecture import ArchitectureRegistry
    from radjax_student.architecture.rwkv7_reference import (
        RWKV7_REFERENCE_ARCHITECTURE_ID,
        RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        reference_architecture_config,
        register_rwkv7_reference,
    )
    from radjax_student.contracts import ObjectiveConfig, ObjectiveScope, UpdateScope
    from radjax_student.learning import (
        JaxLearningAssemblyRegistries,
        JaxLearningAssemblyRequest,
        LearningState,
        assemble_jax_learning_lifecycle,
    )
    from radjax_student.objectives import (
        SPARSE_CROSS_ENTROPY_IDENTITY,
        build_default_objective_registry,
    )
    from radjax_student.optimizers import (
        OptimizerConfig,
        OptimizerRegistry,
        SgdOptimizer,
    )
    from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry

    architectures = ArchitectureRegistry()
    register_rwkv7_reference(architectures)
    optimizers = OptimizerRegistry()
    optimizers.register(SgdOptimizer())
    request = JaxLearningAssemblyRequest(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        architecture_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        architecture_config=reference_architecture_config(),
        objective_identity=SPARSE_CROSS_ENTROPY_IDENTITY,
        objective_config=ObjectiveConfig(
            SPARSE_CROSS_ENTROPY_IDENTITY, {"reduction": "mean"}
        ),
        optimizer_id="sgd.v1",
        optimizer_version=1,
        optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.05),
        runtime_backend_id="jax",
        runtime_implementation_version="p2.9",
        runtime_config=RuntimeConfig(
            backend_id="jax",
            platform_preference="cpu",
            precision_policy="float32",
            placement_policy="single_device",
            compilation_policy=compilation_policy,
            distributed_policy="disabled",
            fallback_policy="disallowed",
            seed=17,
        ),
        root_seed=17,
        learning_state=LearningState(
            "p48-rwkv7",
            active_update_scope=UpdateScope("whole_student"),
            active_objective_scope=ObjectiveScope(),
        ),
    )
    return assemble_jax_learning_lifecycle(
        request,
        registries=JaxLearningAssemblyRegistries(
            architectures,
            build_default_objective_registry(),
            optimizers,
            build_default_runtime_registry(),
        ),
    )


def _batch(tokens: tuple[int, ...] = _TOKENS) -> Any:
    from radjax_student.learning import LearningBatch

    return LearningBatch(
        "p48-rwkv7",
        inputs={"token_ids": [list(tokens)]},
        targets={"token_ids": [list(_TARGETS)]},
    )


def _execute(assembly: Any, batch: Any) -> Any:
    lifecycle = assembly.loop_executor.lifecycle
    return assembly.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        parameters=lifecycle.parameters,
        objective=lifecycle.objective_selection,
        batch=batch,
    )


def _all_finite(jax: Any, jnp: Any, value: Any) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(value)
    )


def _changed(jax: Any, jnp: Any, first: Any, second: Any) -> bool:
    return any(
        not bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree_util.tree_leaves(first),
            jax.tree_util.tree_leaves(second),
            strict=True,
        )
    )


def _allclose(jax: Any, jnp: Any, first: Any, second: Any) -> bool:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    return first_tree == second_tree and all(
        bool(jnp.allclose(left, right, rtol=1e-5, atol=2e-5))
        for left, right in zip(first_leaves, second_leaves, strict=True)
    )


def _forward(jax: Any, jnp: Any, lifecycle: Any) -> Any:
    from radjax_student.learning.jax_batch import FiniteJsonJaxBatchMaterializer

    materialized = FiniteJsonJaxBatchMaterializer().materialize(_batch())
    return lifecycle.architecture.apply_jax(
        jax.tree_util.tree_map(jnp.asarray, lifecycle.parameters),
        jax.tree_util.tree_map(jnp.asarray, lifecycle.architecture_carry),
        materialized,
        objective_scope=lifecycle.learning_state.active_objective_scope,
        training=False,
        rng_key=None,
    )


def _fixture_provenance(root: Path) -> dict[str, Any]:
    fixture_path = root / "tests/fixtures/rwkv7_reference/parity_fixture.json"
    provenance_path = root / "tests/fixtures/rwkv7_reference/provenance.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    verified = provenance["fixture"]["sha256"] == _hash(fixture_path)
    return {
        "verified": verified,
        "pinned_source": fixture["pinned_source"],
        "fixture_sha256": _hash(fixture_path),
        "provenance_sha256": _hash(provenance_path),
        "generator": provenance["generator"],
        "oracle": provenance["oracle"],
        "domain": fixture["domain"],
        "tolerance": {"rtol": 1e-5, "atol": 2e-5},
    }


def generate_phase4_report(root: Path, workdir: Path) -> dict[str, Any]:
    """Execute the fixed P4.4--P4.7 path and return typed P4.8 evidence."""

    import jax
    import jax.numpy as jnp

    from radjax_student.checkpoints import save_learning_checkpoint_v3

    root = root.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    neutrality = build_phase4_architecture_ingestion_audit(root)
    if neutrality["status"] != "pass":
        raise ValueError("phase4_architecture_neutrality_blocked")

    eager = _assembly("eager")
    jit = _assembly("jit")
    eager_before, jit_before = eager.lifecycle, jit.lifecycle
    eager_execution, jit_execution = _execute(eager, _batch()), _execute(jit, _batch())
    eager_after, jit_after = eager.loop_executor.lifecycle, jit.loop_executor.lifecycle

    with tempfile.TemporaryDirectory(dir=workdir) as directory:
        checkpoint = Path(directory) / "checkpoint"
        save_learning_checkpoint_v3(
            eager_after.checkpoint(), checkpoint, optimizer=eager_after.optimizer
        )
        manifest_digest = _hash(checkpoint / "manifest.json")
        hf_descriptor_digest = _hash(checkpoint / "hf_descriptor.json")
        restored_assembly = _assembly("eager")
        restored = restored_assembly.lifecycle.restore_from_checkpoint(checkpoint)
        source_forward, restored_forward = (
            _forward(jax, jnp, eager_after),
            _forward(jax, jnp, restored),
        )
        restored_assembly.loop_executor.lifecycle = restored
        source_next = _execute(eager, _batch(_NEXT_TOKENS))
        restored_next = _execute(restored_assembly, _batch(_NEXT_TOKENS))
        source_after = eager.loop_executor.lifecycle
        restored_after = restored_assembly.loop_executor.lifecycle

    summary = eager.summary
    jit_forward = _forward(jax, jnp, jit_after)
    eager_loss_finite = eager_execution.result.loss is not None and math.isfinite(
        eager_execution.result.loss.loss
    )
    eager_gradient_finite = _all_finite(jax, jnp, eager_execution.gradients)
    jit_loss_finite = jit_execution.result.loss is not None and math.isfinite(
        jit_execution.result.loss.loss
    )
    jit_gradient_finite = _all_finite(jax, jnp, jit_execution.gradients)
    lifecycle = {
        "eager": {
            "forward_finite": _all_finite(jax, jnp, source_forward.outputs),
            "loss_finite": eager_loss_finite,
            "gradient_finite": eager_gradient_finite,
            "finite": eager_loss_finite and eager_gradient_finite,
            "parameters_changed": _changed(
                jax, jnp, eager_before.parameters, eager_after.parameters
            ),
            "carry_changed": _changed(
                jax,
                jnp,
                eager_before.architecture_carry,
                eager_after.architecture_carry,
            ),
            "optimizer_advanced": eager_after.optimizer_state.envelope.step == 1,
            "key_advanced": eager_execution.runtime_result.output_metadata["rng_bridge"]
            == {
                "schema_version": "runtime_jax_key_bridge.v1",
                "prng_implementation": "threefry2x32",
                "stream": "dropout",
                "slot": "dropout",
                "global_step": 0,
                "micro_step": 0,
                "invocation_index": 0,
            },
        },
        "jit": {
            "forward_finite": _all_finite(jax, jnp, jit_forward.outputs),
            "loss_finite": jit_loss_finite,
            "gradient_finite": jit_gradient_finite,
            "finite": jit_loss_finite and jit_gradient_finite,
            "parameters_changed": _changed(
                jax, jnp, jit_before.parameters, jit_after.parameters
            ),
            "carry_changed": _changed(
                jax, jnp, jit_before.architecture_carry, jit_after.architecture_carry
            ),
            "compiled": jit_execution.runtime_result.compiled,
        },
        "eager_jit_parameter_equality": _allclose(
            jax, jnp, eager_after.parameters, jit_after.parameters
        ),
        "prepared_identities": {
            "eager": eager_execution.runtime_result.prepared_execution_digest,
            "jit": jit_execution.runtime_result.prepared_execution_digest,
            "distinct": (
                eager_execution.runtime_result.prepared_execution_digest
                != jit_execution.runtime_result.prepared_execution_digest
            ),
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "plugin": {
            "architecture_id": eager_after.architecture.architecture_id,
            "architecture_version": eager_after.architecture.architecture_version,
        },
        "identities": {
            "architecture_config_digest": eager_after.config_digest,
            "parameter_catalog_digest": eager_after.catalog_digest,
            "parameter_layout_digest": eager_after.parameter_layout.digest(),
            "carry_descriptor_digest": eager_after.architecture_carry_descriptor[
                "pytree_descriptor_digest"
            ],
            "hf_descriptor_digest": eager_after.hf_descriptor.digest,
            "hf_reference": eager_after.hf_reference.to_dict(),
        },
        "runtime_callable": {
            "callable_id": summary["runtime_callable_id"],
            "callable_version": summary["runtime_callable_version"],
            "identity_digest": summary["runtime_callable_identity_digest"],
        },
        "lifecycle": lifecycle,
        "checkpoint_replay": {
            "manifest_digest": manifest_digest,
            "hf_descriptor_file_digest": hf_descriptor_digest,
            "restored_forward_equal": _allclose(
                jax, jnp, source_forward.outputs, restored_forward.outputs
            ),
            "restored_carry_equal": _allclose(
                jax,
                jnp,
                source_forward.updated_architecture_carry,
                restored_forward.updated_architecture_carry,
            ),
            "next_step_loss_equal": (
                source_next.result.loss is not None
                and restored_next.result.loss is not None
                and math.isclose(
                    source_next.result.loss.loss,
                    restored_next.result.loss.loss,
                    rel_tol=1e-5,
                    abs_tol=2e-5,
                )
            ),
            "next_step_parameters_equal": _allclose(
                jax, jnp, source_after.parameters, restored_after.parameters
            ),
            "next_step_carry_equal": _allclose(
                jax,
                jnp,
                source_after.architecture_carry,
                restored_after.architecture_carry,
            ),
        },
        "architecture_neutrality": {
            "schema_version": neutrality["schema_version"],
            "status": neutrality["status"],
            "audit_digest": _digest(neutrality),
            "reviewed_source_count": len(neutrality["reviewed_source_paths"]),
            "approved_generic_changes": neutrality["approved_generic_changes"],
        },
        "fixture_provenance": {
            **_fixture_provenance(root),
            "report_generator": {
                "path": (
                    "src/radjax_student/validation/p4_8_architecture_ingestion/"
                    "runner_jax.py"
                ),
                "sha256": _hash(Path(__file__)),
            },
        },
        "equation_parity_claim": EQUATION_PARITY_CLAIM,
        "initialization_parity_claim": NON_CLAIM,
        "training_recipe_parity_claim": NON_CLAIM,
        "weight_file_compatibility": False,
        "non_claims": {name: NON_CLAIM for name in sorted(REQUIRED_NON_CLAIMS)},
    }
    report["status"] = derive_status(report)
    report["evidence_digest"] = _digest(report)
    canonical_report_bytes(report)
    return report


def write_phase4_report(root: Path, workdir: Path, output: Path) -> Path:
    """Write one canonical report while keeping execution artifacts temporary."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_report_bytes(generate_phase4_report(root, workdir)))
    return output
