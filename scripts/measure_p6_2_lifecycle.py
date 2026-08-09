"""Measure one admitted v6 behavioral source unit through the generic JAX path."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from hashlib import sha256
from pathlib import Path
from typing import Any


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _digest(value: object) -> str:
    return (
        "sha256:"
        + sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _first_training_coordinate(batches: Any) -> tuple[str, int, int]:
    corridor = batches.training_corridor
    rows = tuple(sorted({int(value) for value in corridor.example_indices}))
    identifiers = dict(zip(rows, corridor.example_ids, strict=True))
    index = 0
    return (
        identifiers[int(corridor.example_indices[index])],
        int(corridor.positions[index]),
        int(corridor.mode_ids[index]),
    )


def _array_bytes(value: object, seen: set[int] | None = None) -> int:
    """Count unique host arrays retained by a value without inferring devices."""

    seen = set() if seen is None else seen
    marker = id(value)
    if marker in seen:
        return 0
    seen.add(marker)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if dataclasses.is_dataclass(value):
        return sum(
            _array_bytes(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
        )
    if isinstance(value, dict):
        return sum(_array_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(item, seen) for item in value)
    return 0


def _projection_resource_bytes(projection: Any) -> int:
    """Count the public resource bytes consumed by P5.4 materialization."""

    return sum(
        int(resource.raw_size_bytes)
        for resource in (
            projection.authority_reference,
            projection.corridor_mode_table,
            *projection.target_shard.components.values(),
            *projection.corridor_assignment.components.values(),
        )
    )


def _registries() -> Any:
    from radjax_student.architecture import ArchitectureRegistry
    from radjax_student.architecture.rwkv7_reference import register_rwkv7_reference
    from radjax_student.behavior import BehaviorJaxBatchMaterializerV1
    from radjax_student.learning import JaxLearningAssemblyRegistries
    from radjax_student.objectives import build_default_objective_registry
    from radjax_student.optimizers import OptimizerRegistry, SgdOptimizer
    from radjax_student.runtime import build_default_runtime_registry

    architectures = ArchitectureRegistry()
    register_rwkv7_reference(architectures)
    optimizers = OptimizerRegistry()
    optimizers.register(SgdOptimizer())
    return JaxLearningAssemblyRegistries(
        architectures,
        build_default_objective_registry(),
        optimizers,
        build_default_runtime_registry(),
        {"behavior_jax_batch_materializer.v1": BehaviorJaxBatchMaterializerV1()},
    )


def _current_rss_bytes() -> int:
    """Read current process RSS without recording a host-specific identity."""

    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip()) * 1024


def _device_memory_stats() -> dict[str, int] | None:
    """Read official JAX device allocation counters when available."""

    try:
        import jax

        devices = jax.devices()
        if len(devices) != 1:
            return None
        stats = devices[0].memory_stats()
    except (ImportError, AttributeError, RuntimeError):
        return None
    if not stats:
        return None
    return {
        str(key): int(value)
        for key, value in stats.items()
        if isinstance(value, (int, float))
    }


def _device_memory_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    """Summarize official device allocation counters observed at each phase."""

    observations: list[dict[str, int]] = []
    for sample in samples:
        residency = sample.get("residency", {})
        if not isinstance(residency, dict):
            continue
        phases = residency.get("phase_device_memory_stats", {})
        if not isinstance(phases, dict):
            continue
        for value in phases.values():
            if isinstance(value, dict):
                observations.append(
                    {
                        str(key): int(number)
                        for key, number in value.items()
                        if isinstance(number, (int, float))
                    }
                )
    if not observations:
        return {"available": False, "observation_count": 0}
    maxima = {
        key: max(row.get(key, 0) for row in observations)
        for key in {key for row in observations for key in row}
    }
    return {
        "available": True,
        "observation_count": len(observations),
        "bytes_limit": max(row.get("bytes_limit", 0) for row in observations),
        "observed_peak_bytes_in_use": maxima.get("peak_bytes_in_use", 0),
        "observed_peak_bytes_reserved": maxima.get("peak_bytes_reserved", 0),
        "observed_peak_pool_bytes": maxima.get("peak_pool_bytes", 0),
        "maximum_bytes_in_use": maxima.get("bytes_in_use", 0),
        "maximum_pool_bytes": maxima.get("pool_bytes", 0),
    }


def _run(
    artifact: Path,
    *,
    surface: str,
    assembled: Any | None,
    pre_cycle_rss: int,
    checkpoint_restore: bool,
    platform_preference: str,
    compilation_policy: str,
) -> tuple[dict[str, object], Any]:
    from radjax_student.architecture.rwkv7_reference import (
        RWKV7_REFERENCE_ARCHITECTURE_ID,
        RWKV7_REFERENCE_ARCHITECTURE_VERSION,
        configurable_architecture_config,
    )
    from radjax_student.artifacts.native_v3_v6 import (
        open_native_v3_v6_behavioral_projection,
    )
    from radjax_student.behavior import (
        corridor_source_unit_learning_batch_v1,
        exemplar_source_unit_learning_batch_v1,
        materialize_behavioral_batches_v1,
    )
    from radjax_student.checkpoints import save_learning_checkpoint_v3
    from radjax_student.contracts import ObjectiveConfig, ObjectiveScope, UpdateScope
    from radjax_student.learning import (
        JaxLearningAssemblyRequest,
        LearningState,
        assemble_jax_learning_lifecycle,
    )
    from radjax_student.objectives.behavioral import (
        BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY,
        BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY,
        BEHAVIORAL_OBJECTIVE_CONFIG,
    )
    from radjax_student.optimizers import OptimizerConfig
    from radjax_student.runtime import RuntimeConfig

    phase_rss: dict[str, int] = {"pre_cycle": pre_cycle_rss}
    phase_device_memory = {"pre_cycle": _device_memory_stats()}
    admitted_at = time.perf_counter()
    phase_rss["admission_start"] = _current_rss_bytes()
    projection = open_native_v3_v6_behavioral_projection(artifact)
    batches = materialize_behavioral_batches_v1(projection)
    phase_rss["admission_materialization_end"] = _current_rss_bytes()
    if surface == "corridor":
        source_unit = corridor_source_unit_learning_batch_v1(
            batches,
            partition="training",
            coordinate=_first_training_coordinate(batches),
        )
        objective_identity = BEHAVIORAL_CORRIDOR_OBJECTIVE_IDENTITY
    elif surface == "exemplar":
        passport = batches.training_exemplars.passports[0]
        source_unit = exemplar_source_unit_learning_batch_v1(
            batches,
            partition="training",
            passport_key=(
                str(passport["selected_example_id"]),
                int(passport["selected_position"]),
                str(passport["corridor_fingerprint_id"]),
            ),
        )
        objective_identity = BEHAVIORAL_EXEMPLAR_OBJECTIVE_IDENTITY
    else:  # pragma: no cover - argparse supplies the closed surface set.
        raise ValueError("surface must be corridor or exemplar")
    admission_materialization_seconds = time.perf_counter() - admitted_at
    language = projection.language
    config = configurable_architecture_config(
        language.vocabulary.vocabulary_size,
        len(source_unit.inputs["token_ids"][0]),
        tokenizer=language.tokenizer,
        vocabulary=language.vocabulary,
        special_tokens=language.special_tokens,
    )
    assembly_started = time.perf_counter()
    phase_rss["assembly_start"] = _current_rss_bytes()
    if assembled is None:
        assembled = assemble_jax_learning_lifecycle(
            JaxLearningAssemblyRequest(
                architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
                architecture_version=RWKV7_REFERENCE_ARCHITECTURE_VERSION,
                architecture_config=config,
                objective_identity=objective_identity,
                objective_config=ObjectiveConfig(
                    objective_identity, BEHAVIORAL_OBJECTIVE_CONFIG
                ),
                optimizer_id="sgd.v1",
                optimizer_version=1,
                optimizer_config=OptimizerConfig("sgd.v1", learning_rate=0.01),
                runtime_backend_id="jax",
                runtime_implementation_version="p2.9",
                runtime_config=RuntimeConfig(
                    backend_id="jax",
                    platform_preference=platform_preference,
                    precision_policy="float32",
                    placement_policy="single_device",
                    compilation_policy=compilation_policy,
                    distributed_policy="disabled",
                    fallback_policy="disallowed",
                    seed=62,
                ),
                root_seed=62,
                learning_state=LearningState(
                    "p6-2-v6-behavioral-lifecycle",
                    active_update_scope=UpdateScope(),
                    active_objective_scope=ObjectiveScope(),
                ),
                batch_materializer_id="behavior_jax_batch_materializer.v1",
            ),
            registries=_registries(),
        )
    elif assembled.lifecycle.architecture_config != config:
        raise RuntimeError("P6.2 repetition changed the accepted configuration")
    assembly_seconds = time.perf_counter() - assembly_started
    phase_rss["assembly_end"] = _current_rss_bytes()
    lifecycle = assembled.loop_executor.lifecycle
    transfer_at = time.perf_counter()
    phase_rss["execution_start"] = _current_rss_bytes()
    materialized = assembled.loop_executor.batch_materializer.materialize(source_unit)
    import jax

    for leaf in jax.tree_util.tree_leaves(
        (materialized.inputs, materialized.targets, materialized.weights)
    ):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    source_to_jax_transfer_seconds = time.perf_counter() - transfer_at
    model_at = time.perf_counter()
    forward = lifecycle.architecture.apply_jax(
        lifecycle.parameters,
        lifecycle.architecture_carry,
        materialized,
        architecture_config=lifecycle.architecture_config,
        objective_scope=lifecycle.learning_state.active_objective_scope,
        training=False,
        rng_key=None,
    )
    for leaf in jax.tree_util.tree_leaves(
        (forward.outputs, forward.updated_architecture_carry)
    ):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    rwkv_forward_seconds = time.perf_counter() - model_at
    tracemalloc.start()
    allocation_before = tracemalloc.take_snapshot()
    executed_at = time.perf_counter()
    execution = assembled.loop_executor(
        architecture=lifecycle.architecture,
        architecture_config=lifecycle.architecture_config,
        optimizer=lifecycle.optimizer,
        optimizer_config=lifecycle.optimizer_config,
        optimizer_state=lifecycle.optimizer_state,
        learning_state=lifecycle.learning_state,
        objective=lifecycle.objective_selection,
        batch=source_unit,
    )
    execution_seconds = time.perf_counter() - executed_at
    phase_rss["execution_end"] = _current_rss_bytes()
    phase_device_memory["execution_end"] = _device_memory_stats()
    allocation_after = tracemalloc.take_snapshot()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = assembled.loop_executor.lifecycle
    checkpointed_at = time.perf_counter()
    phase_rss["checkpoint_restore_start"] = _current_rss_bytes()
    if checkpoint_restore:
        with tempfile.TemporaryDirectory(prefix="radjax-p6-2-") as temporary:
            destination = Path(temporary) / "checkpoint"
            saved = save_learning_checkpoint_v3(
                after.checkpoint(), destination, optimizer=after.optimizer
            )
            restored = after.restore_from_checkpoint(destination)
        checkpoint_roundtrip_seconds = time.perf_counter() - checkpointed_at
        checkpoint_identity = dict(saved.integrity)
        checkpoint_restore_config_digest = restored.config_digest
    else:
        checkpoint_roundtrip_seconds = 0.0
        checkpoint_identity = None
        checkpoint_restore_config_digest = after.config_digest
    phase_rss["checkpoint_restore_end"] = _current_rss_bytes()
    phase_device_memory["checkpoint_restore_end"] = _device_memory_stats()
    if execution.runtime_result.status != "pass" or execution.result.loss is None:
        raise RuntimeError("P6.2 lifecycle did not report a successful loss")
    if not execution.result.changed_parameter_paths:
        raise RuntimeError("P6.2 lifecycle did not report parameter movement")
    if checkpoint_restore and checkpoint_restore_config_digest != after.config_digest:
        raise RuntimeError("P6.2 checkpoint restore lost architecture configuration")
    return {
        "schema_version": "radjax.phase6.p6_2_lifecycle_measurement.v1",
        "artifact_transport": "archive" if artifact.suffix == ".tgz" else "directory",
        "behavioral_source_identity": projection.behavioral_source_identity,
        "split_identity": batches.split.split_identity,
        "source_unit_identity": source_unit.metadata["source_unit_identity"],
        "architecture_config_digest": after.config_digest,
        "parameter_layout_digest": after.parameter_layout.digest(),
        "runtime_callable": execution.runtime_result.callable_reference.callable_id,
        "batch_materializer": assembled.summary["batch_materializer_identity"],
        "objective_identity": objective_identity.to_dict(),
        "loss": float(execution.result.loss.loss),
        "changed_parameter_count": len(execution.result.changed_parameter_paths),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_restore_config_digest": checkpoint_restore_config_digest,
        "checkpoint_restore_performed": checkpoint_restore,
        "timing_seconds": {
            "admission_materialization": admission_materialization_seconds,
            "assembly": assembly_seconds,
            "generic_lifecycle": execution_seconds,
            "rwkv_forward": rwkv_forward_seconds,
            "source_to_jax_transfer": source_to_jax_transfer_seconds,
            "checkpoint_roundtrip": checkpoint_roundtrip_seconds,
        },
        "residency": {
            "phase_rss_bytes": phase_rss,
            "peak_current_rss_bytes": max(phase_rss.values()),
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "retained_materialization_array_bytes": _array_bytes(batches),
            "p5_materialization_resource_bytes": _projection_resource_bytes(projection),
            "python_tracemalloc_peak_bytes_during_execution": traced_peak,
            "python_allocation_count_during_execution": sum(
                max(stat.count_diff, 0)
                for stat in allocation_after.compare_to(allocation_before, "lineno")
            ),
            "phase_device_memory_stats": phase_device_memory,
        },
        "runtime_event": {
            "mode": execution.runtime_result.mode,
            "compiled": execution.runtime_result.compiled,
            "compilation_seconds": execution.runtime_result.compilation_seconds,
            "preparation_seconds": execution.runtime_result.preparation_seconds,
            "dispatch_seconds": execution.runtime_result.dispatch_seconds,
            "synchronization_seconds": execution.runtime_result.synchronization_seconds,
            "callable_id": execution.runtime_result.callable_reference.callable_id,
            "prepared_execution_digest": (
                execution.runtime_result.prepared_execution_digest
            ),
        },
        "throughput": {
            "source_units_per_second": 1.0 / execution_seconds,
            "coordinate": dict(source_unit.metadata["coordinate"]),
        },
        "nonclaims": [
            "no_batched_rwkv_support",
            "no_structural_rwkv_generalization",
            "no_accelerator_measurement",
        ],
    }, assembled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface", choices=("corridor", "exemplar"), required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--platform-preference", choices=("cpu", "gpu", "tpu"), default="cpu"
    )
    parser.add_argument(
        "--compilation-policy", choices=("eager", "jit"), default="eager"
    )
    parser.add_argument(
        "--skip-checkpoint-restore",
        action="store_true",
        help="diagnostic execution-only run; not the accepted full-cycle workload",
    )
    args = parser.parse_args()
    if not args.artifact.exists():
        raise SystemExit("artifact must exist")
    if args.repetitions < 1:
        raise SystemExit("repetitions must be at least one")
    samples: list[dict[str, object]] = []
    release_rss_bytes: list[int] = []
    assembled: Any | None = None
    for _ in range(args.repetitions):
        pre_cycle_rss = _current_rss_bytes()
        sample, assembled = _run(
            args.artifact,
            surface=args.surface,
            assembled=assembled,
            pre_cycle_rss=pre_cycle_rss,
            checkpoint_restore=not args.skip_checkpoint_restore,
            platform_preference=args.platform_preference,
            compilation_policy=args.compilation_policy,
        )
        samples.append(sample)
        gc.collect()
        release_rss_bytes.append(_current_rss_bytes())
    continuity_fields = (
        "behavioral_source_identity",
        "split_identity",
        "architecture_config_digest",
        "parameter_layout_digest",
        "runtime_callable",
        "batch_materializer",
        "objective_identity",
        "source_unit_identity",
        "checkpoint_restore_config_digest",
    )
    if any(
        sample[field] != samples[0][field]
        for sample in samples[1:]
        for field in continuity_fields
    ):
        raise RuntimeError("P6.2 repetitions lost identity continuity")
    payload = {
        "schema_version": "radjax.phase6.p6_2_lifecycle_measurement.v2",
        "artifact_transport": (
            "archive" if args.artifact.suffix == ".tgz" else "directory"
        ),
        "surface": args.surface,
        "checkpoint_restore_performed": not args.skip_checkpoint_restore,
        "repetitions": args.repetitions,
        "platform_preference": args.platform_preference,
        "compilation_policy": args.compilation_policy,
        "release_rss_bytes_after_gc": release_rss_bytes,
        "rss_summary": {
            "pre_cycle_bytes": [
                sample["residency"]["phase_rss_bytes"]["pre_cycle"]
                for sample in samples
            ],
            "peak_current_bytes": [
                sample["residency"]["peak_current_rss_bytes"] for sample in samples
            ],
            "post_release_gc_bytes": release_rss_bytes,
        },
        "device_memory_summary": _device_memory_summary(samples),
        "identity_continuity": {
            field: samples[0][field] for field in continuity_fields
        },
        "runtime_event_continuity": {
            "callable_id": samples[0]["runtime_event"]["callable_id"],
            "mode": samples[0]["runtime_event"]["mode"],
            "prepared_execution_digest": samples[0]["runtime_event"][
                "prepared_execution_digest"
            ],
        },
        "samples": samples,
        "nonclaims": samples[0]["nonclaims"],
    }
    payload["measurement_identity"] = _digest(
        {key: value for key, value in payload.items() if key != "timing_seconds"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
