"""Execute the accepted P6.6 reduced-burn schedule through generic JAX."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

from radjax_student.artifacts.native_v3_v6 import (
    open_native_v3_v6_behavioral_projection,
)
from radjax_student.behavior import materialize_behavioral_batches_v1
from radjax_student.behavior.learning_batch import (
    corridor_source_unit_learning_batch_v1,
    exemplar_source_unit_learning_batch_v1,
)

HOST_CEILING = 23_284_565_606
DEVICE_LIMIT = 11_727_028_224
CORRIDOR_EPOCHS = 64
EXEMPLAR_EPOCHS = 64
CHECKPOINT_INTERVAL = 8


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
    )


def _source_units(batches, surface: str):
    if surface == "corridor":
        batch = batches.training_corridor
        sequence_length = batch.input_ids.shape[1]
        coordinates = []
        for row, position in enumerate(batch.positions):
            if int(position) != 0:
                continue
            example_row = row // sequence_length
            coordinates.append(
                (
                    str(batch.example_ids[example_row]),
                    int(position),
                    int(batch.mode_ids[row]),
                )
            )
        coordinates.sort()
        return [
            (
                corridor_source_unit_learning_batch_v1(
                    batches, partition="training", coordinate=coordinate
                ),
                coordinate,
            )
            for coordinate in coordinates
        ]
    passports = sorted(
        (
            str(item["selected_example_id"]),
            int(item["selected_position"]),
            str(item["corridor_fingerprint_id"]),
        )
        for item in batches.training_exemplars.passports
    )
    return [
        (
            exemplar_source_unit_learning_batch_v1(
                batches, partition="training", passport_key=passport
            ),
            passport,
        )
        for passport in passports
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-preference", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--compilation-policy", choices=("eager", "jit"), default="jit")
    parser.add_argument(
        "--surface", choices=("corridor", "exemplar", "both"), default="both"
    )
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    from measure_p6_2_lifecycle import _current_rss_bytes, _run

    started = time.perf_counter()
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    projection = open_native_v3_v6_behavioral_projection(args.artifact)
    batches = materialize_behavioral_batches_v1(projection)
    surfaces = ("corridor", "exemplar") if args.surface == "both" else (args.surface,)
    passes = []
    blocked = None
    for surface in surfaces:
        units = _source_units(batches, surface)
        epochs = CORRIDOR_EPOCHS if surface == "corridor" else EXEMPLAR_EPOCHS
        declared_scheduled_items = len(units) * epochs
        scheduled = units * epochs
        if args.max_items is not None:
            scheduled = scheduled[: args.max_items]
        pass_started = time.perf_counter()
        assembled = None
        records = []
        checkpoint_count = 0
        for index, (source_unit, coordinate) in enumerate(scheduled):
            cycle_started = time.perf_counter()
            try:
                sample, assembled = _run(
                    args.artifact,
                    surface=surface,
                    assembled=assembled,
                    pre_cycle_rss=_current_rss_bytes(),
                    checkpoint_restore=(index + 1) % CHECKPOINT_INTERVAL == 0,
                    platform_preference=args.platform_preference,
                    compilation_policy=args.compilation_policy,
                    source_unit_override=source_unit,
                    projection_override=projection,
                    batches_override=batches,
                )
            except Exception as exc:  # preserve exact failure boundary
                blocked = {
                    "surface": surface,
                    "cursor": index,
                    "scheduled_items": len(scheduled),
                    "declared_scheduled_items": declared_scheduled_items,
                    "failure_type": type(exc).__name__,
                    "failure": str(exc),
                }
                break
            if sample["checkpoint_restore_performed"]:
                checkpoint_count += 1
            record = {
                "cursor": index + 1,
                "epoch": (index // len(units)),
                "within_epoch_index": index % len(units),
                "coordinate": coordinate,
                "cycle_seconds": time.perf_counter() - cycle_started,
                "loss": sample["loss"],
                "checkpoint_restore_performed": sample["checkpoint_restore_performed"],
                "checkpoint_identity": sample["checkpoint_identity"],
                "runtime_event": sample["runtime_event"],
                "throughput": sample["throughput"],
                "timing_seconds": sample["timing_seconds"],
                "residency": sample["residency"],
            }
            records.append(record)
            progress_path.write_text(
                json.dumps(
                    {
                        "surface": surface,
                        "cursor": index + 1,
                        "declared_scheduled_items": declared_scheduled_items,
                        "host_rss_bytes": record["residency"]["process_peak_rss_bytes"],
                        "device_memory": record["residency"].get(
                            "phase_device_memory_stats", {}
                        ),
                        "compiled": record["runtime_event"]["compiled"],
                        "compilation_seconds": record["runtime_event"][
                            "compilation_seconds"
                        ],
                        "checkpoint": record["checkpoint_restore_performed"],
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if record["residency"]["process_peak_rss_bytes"] >= HOST_CEILING or any(
                value is not None and value.get("peak_pool_bytes", 0) > DEVICE_LIMIT
                for value in record["residency"]
                .get("phase_device_memory_stats", {})
                .values()
            ):
                blocked = {
                    "surface": surface,
                    "cursor": index + 1,
                    "scheduled_items": len(scheduled),
                    "failure_type": "resource_envelope_exceeded",
                    "failure": "frozen P6.5 host/device envelope exceeded",
                }
                break
            if (index + 1) % CHECKPOINT_INTERVAL == 0:
                gc.collect()
        passes.append(
            {
                "surface": surface,
                "source_units_per_epoch": len(units),
                "epochs": epochs,
                "scheduled_items": len(scheduled),
                "declared_scheduled_items": declared_scheduled_items,
                "completed_items": len(records),
                "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
                "checkpoint_count": checkpoint_count,
                "completion_seconds": time.perf_counter() - pass_started,
                "records": records,
                "runtime_callable": (
                    records[0]["runtime_event"]["callable_id"] if records else None
                ),
                "prepared_execution_digests": sorted(
                    {
                        record["runtime_event"]["prepared_execution_digest"]
                        for record in records
                    }
                ),
                "compile_events": sum(
                    bool(record["runtime_event"]["compiled"]) for record in records
                ),
                "compilation_seconds": sum(
                    record["runtime_event"]["compilation_seconds"] for record in records
                ),
                "throughput_source_units_per_second": (
                    len(records) / max(time.perf_counter() - pass_started, 1e-9)
                ),
            }
        )
        if blocked:
            break
    payload = {
        "schema_version": "radjax.student.p6_6_reduced_burn.v1",
        "authority": {
            "student_commit": "6c0d99435f5a8b4a38212d7050a73ec60bd6509a",
            "tome_commit": "6a6c65378cfd86a190e44e861ed9323927c2acc8",
            "contract_release": "0.9.0",
            "contract_commit": "1fa43e1aea2e198511db86dafb0aeefa525d48c7",
            "artifact": str(args.artifact),
        },
        "policy": {
            "source_batch_size": 1,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "carry_policy": "reset_each_independent_example.v1",
            "corridor_epochs": CORRIDOR_EPOCHS,
            "exemplar_epochs": EXEMPLAR_EPOCHS,
            "host_ceiling_bytes": HOST_CEILING,
            "device_limit_bytes": DEVICE_LIMIT,
        },
        "platform_preference": args.platform_preference,
        "compilation_policy": args.compilation_policy,
        "passes": passes,
        "blocked": blocked,
        "completion_seconds": time.perf_counter() - started,
        "classification": (
            "resource_envelope_blocked" if blocked else "minimum_burn_completed"
        ),
        "limitations": [
            "p6_5_execution_retention_observed",
            "p6_5_repeated_jit_compilation_observed",
            "no_allocator_or_memory_management_change",
        ],
    }
    payload["measurement_identity"] = _digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if blocked is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
