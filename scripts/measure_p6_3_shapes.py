"""Measure equation-authorized RWKV shape rehearsals through generic seams."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from radjax_student.architecture import ArchitectureInitRequest, ArchitectureRegistry
from radjax_student.architecture.rwkv7_reference import (
    RWKV7ReferencePlugin,
    configurable_architecture_config,
    register_rwkv7_reference,
)
from radjax_student.checkpoints import save_learning_checkpoint_v3
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    LearningBatch,
    ObjectiveConfig,
    ObjectiveScope,
    UpdateScope,
    hf_digest,
)
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
from radjax_student.optimizers import OptimizerConfig, OptimizerRegistry, SgdOptimizer
from radjax_student.runtime import RuntimeConfig, build_default_runtime_registry


def _digest(value: object) -> str:
    return (
        "sha256:"
        + sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _config(spec: dict[str, int]):
    vocabulary_size = spec["vocabulary_size"]
    context_length = spec["context_length"]
    return configurable_architecture_config(
        vocabulary_size,
        context_length,
        tokenizer=HFTokenizerIdentity(
            "p6-3-measurement-tokenizer",
            "v1",
            hf_digest({"vocabulary_size": vocabulary_size}),
            hf_digest({"context_length": context_length}),
            "synthetic",
            hf_digest({"normalization": "identity"}),
            "synthetic",
        ),
        vocabulary=HFVocabularyIdentity(
            vocabulary_size,
            hf_digest({"vocabulary_size": vocabulary_size}),
            hf_digest({"token_ids": vocabulary_size}),
            hf_digest({"added": []}),
            None,
        ),
        special_tokens=HFSpecialTokenIdentity(0, 1, 2, 3, None),
        **{
            key: value
            for key, value in spec.items()
            if key
            not in {
                "vocabulary_size",
                "context_length",
            }
        },
    )


def _registries() -> JaxLearningAssemblyRegistries:
    architectures = ArchitectureRegistry()
    register_rwkv7_reference(architectures)
    optimizers = OptimizerRegistry()
    optimizers.register(SgdOptimizer())
    return JaxLearningAssemblyRegistries(
        architectures,
        build_default_objective_registry(),
        optimizers,
        build_default_runtime_registry(),
    )


def _array_bytes(value: Any) -> int:
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(item) for item in value)
    return 0


def _run(spec: dict[str, int], *, seed: int) -> dict[str, object]:
    config = _config(spec)
    started = time.perf_counter()
    assembled = assemble_jax_learning_lifecycle(
        JaxLearningAssemblyRequest(
            architecture_id="radjax.architecture.rwkv7_reference",
            architecture_version=1,
            architecture_config=config,
            objective_identity=SPARSE_CROSS_ENTROPY_IDENTITY,
            objective_config=ObjectiveConfig(
                SPARSE_CROSS_ENTROPY_IDENTITY, {"reduction": "mean"}
            ),
            optimizer_id="sgd.v1",
            optimizer_version=1,
            optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.01),
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
                seed=seed,
            ),
            root_seed=seed,
            learning_state=LearningState(
                "p6-3-shape-rehearsal",
                active_update_scope=UpdateScope(),
                active_objective_scope=ObjectiveScope(),
            ),
        ),
        registries=_registries(),
    )
    assembly_seconds = time.perf_counter() - started
    lifecycle = assembled.loop_executor.lifecycle
    context_length = int(config.sequence_length)
    length = min(context_length, 8)
    batch = LearningBatch(
        "p6-3-shape-rehearsal",
        inputs={"token_ids": [list(range(length))]},
        targets={"token_ids": [list(range(1, length)) + [0]]},
    )
    execution_started = time.perf_counter()
    execution = assembled.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        objective=lifecycle.objective_selection,
        batch=batch,
    )
    execution_seconds = time.perf_counter() - execution_started
    if execution.runtime_result.status != "pass":
        raise RuntimeError("shape rehearsal did not complete")
    tokens = jnp.asarray([list(range(length))], dtype=jnp.int32)
    plugin = RWKV7ReferencePlugin()
    initial = plugin.initialize_parameters(
        ArchitectureInitRequest(
            config=config,
            runtime_keys_reference=f"p6-3-shape:{seed}",
            precision_policy="float32",
            runtime_initialization_material=jax.random.key(seed),
        )
    )

    def forward(values: Any) -> Any:
        result = plugin.apply_jax(
            initial.parameters,
            initial.architecture_carry,
            type("Batch", (), {"inputs": {"token_ids": values}})(),
            architecture_config=config,
            objective_scope=ObjectiveScope(),
            training=False,
            rng_key=None,
        )
        return result.outputs

    jit_started = time.perf_counter()
    compiled_forward = jax.jit(forward)
    first = compiled_forward(tokens)
    first.block_until_ready()
    jit_first_seconds = time.perf_counter() - jit_started
    steady_started = time.perf_counter()
    second = compiled_forward(tokens)
    second.block_until_ready()
    jit_steady_seconds = time.perf_counter() - steady_started
    if not bool(jnp.allclose(forward(tokens), second, rtol=1e-5, atol=1e-5)):
        raise RuntimeError("eager/JIT shape rehearsal mismatch")
    checkpoint_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="radjax-p6-3-") as temporary:
        saved = save_learning_checkpoint_v3(
            lifecycle.checkpoint(),
            Path(temporary) / "checkpoint",
            optimizer=lifecycle.optimizer,
        )
        checkpoint_bytes = sum(
            path.stat().st_size for path in Path(temporary).rglob("*") if path.is_file()
        )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    return {
        "shape": dict(spec),
        "architecture_config_digest": lifecycle.config_digest,
        "parameter_layout_digest": lifecycle.parameter_layout.digest(),
        "parameter_count": len(lifecycle.parameter_catalog.paths),
        "parameter_bytes": _array_bytes(lifecycle.parameters),
        "changed_parameter_count": len(execution.result.changed_parameter_paths),
        "assembly_seconds": assembly_seconds,
        "execution_seconds": execution_seconds,
        "eager_runtime": {
            "mode": execution.runtime_result.mode,
            "compiled": execution.runtime_result.compiled,
            "compilation_seconds": execution.runtime_result.compilation_seconds,
            "prepared_execution_digest": (
                execution.runtime_result.prepared_execution_digest
            ),
            "callable_id": execution.runtime_result.callable_reference.callable_id,
        },
        "jit_runtime": {
            "first_seconds": jit_first_seconds,
            "steady_seconds": jit_steady_seconds,
            "recompiled": False,
            "agreement": True,
        },
        "checkpoint": {
            "bytes": checkpoint_bytes,
            "seconds": checkpoint_seconds,
            "integrity": _jsonable(saved.integrity),
        },
        "host_peak_rss_bytes": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin"
            else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    specs = [
        {"vocabulary_size": 512, "context_length": 8},
        {"vocabulary_size": 2048, "context_length": 32},
        {"vocabulary_size": 4096, "context_length": 64},
        {
            "vocabulary_size": 2048,
            "context_length": 32,
            "hidden_size": 16,
            "layer_count": 3,
            "head_count": 4,
            "head_size": 4,
            "ffn_width": 32,
        },
    ]
    rows = [_run(spec, seed=args.seed + index) for index, spec in enumerate(specs)]
    payload = {
        "schema_version": "radjax.phase6.p6_3_shape_rehearsal.v1",
        "authority": {
            "equation_source": (
                "BlinkDL/RWKV-LM@442120a5b40f7d764328bebde94324bc8790806f"
            ),
            "phase4_frozen_shape": {
                "vocabulary_size": 16,
                "context_length": 4,
                "hidden_size": 8,
                "layer_count": 2,
                "head_count": 2,
                "head_size": 4,
                "ffn_width": 16,
            },
        },
        "rehearsals": rows,
    }
    payload["measurement_identity"] = _digest(payload)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
