#!/usr/bin/env python3
"""Build a provenance-preserving P6.5 T4 envelope receipt from raw runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _run_summary(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    rss = payload["rss_summary"]["post_release_gc_bytes"]
    samples = payload["samples"]
    device_rows = []
    for sample in samples:
        phases = sample["residency"].get("phase_device_memory_stats", {})
        device_rows.extend(value for value in phases.values() if value is not None)
    device_limit = max((row.get("bytes_limit", 0) for row in device_rows), default=0)
    device_peak = max(
        (row.get("peak_bytes_in_use", 0) for row in device_rows), default=0
    )
    pool_peak = max((row.get("peak_pool_bytes", 0) for row in device_rows), default=0)
    return {
        "raw_path": str(path),
        "raw_digest": _digest_bytes(raw),
        "surface": payload["surface"],
        "platform_preference": payload["platform_preference"],
        "compilation_policy": payload["compilation_policy"],
        "repetitions": payload["repetitions"],
        "fresh_process": True,
        "host_rss_post_release_gc_bytes": rss,
        "host_rss_first_bytes": rss[0],
        "host_rss_last_bytes": rss[-1],
        "host_rss_peak_bytes": max(rss),
        "device_memory_limit_bytes": device_limit,
        "device_peak_bytes_in_use": device_peak,
        "device_peak_pool_bytes": pool_peak,
        "runtime_event_continuity": payload["runtime_event_continuity"],
        "compiled_events": [sample["runtime_event"]["compiled"] for sample in samples],
        "compilation_seconds": [
            sample["runtime_event"]["compilation_seconds"] for sample in samples
        ],
        "measurement_identity": payload["measurement_identity"],
        "losses": [sample["loss"] for sample in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--diagnostic", action="append", type=Path, default=[])
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()
    runs = [_run_summary(path) for path in args.run]
    diagnostics = [_run_summary(path) for path in args.diagnostic]
    cpu = next((run for run in runs if run["platform_preference"] == "cpu"), None)
    gpu = next(
        (
            run
            for run in runs
            if run["platform_preference"] == "gpu"
            and run["compilation_policy"] == "eager"
            and run["surface"] == "corridor"
        ),
        None,
    )
    numerical = None
    if cpu is not None and gpu is not None:
        deltas = [
            abs(left - right)
            for left, right in zip(cpu["losses"], gpu["losses"], strict=False)
        ]
        numerical = {
            "cpu_run": cpu["raw_path"],
            "gpu_run": gpu["raw_path"],
            "loss_abs_delta_max": max(deltas, default=0.0),
            "declared_float32_tolerance": 1.0e-4,
            "within_declared_tolerance": max(deltas, default=0.0) <= 1.0e-4,
            "note": (
                "CPU comparison is a numerical diagnostic; it is not a T4 "
                "qualification."
            ),
        }
    payload = {
        "schema_version": "radjax.phase6.p6_5_t4_envelope.v1",
        "source_commit": args.source_commit,
        "environment": json.loads(args.environment.read_text(encoding="utf-8")),
        "burn_envelope_policy": json.loads(args.policy.read_text(encoding="utf-8")),
        "runs": runs,
        "diagnostic_runs": diagnostics,
        "numerical_comparison": numerical,
        "commands": args.command,
        "classification": {
            "retention": "execution_retention_observed",
            "owner": "owner_unresolved",
            "scale_impact": "scale_impact_pending_external_envelope",
        },
        "nonclaims": [
            "no_student_owned_retaining_object_demonstrated",
            "jax_allocator_diagnostics_are_not_production_defaults",
            "no_frozen_p6_6_capacity_envelope_was_available",
            "cpu_substitution_is_not_a_t4_qualification",
        ],
    }
    payload["receipt_identity"] = _digest_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
