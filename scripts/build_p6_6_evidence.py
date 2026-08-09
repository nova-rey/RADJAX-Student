"""Write the bounded P6.6 reduced-burn evidence receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_bytes = args.raw.read_bytes()
    raw = json.loads(raw_bytes)
    passes = []
    for value in raw["passes"]:
        records = value["records"]
        rss = [record["residency"]["process_peak_rss_bytes"] for record in records]
        device_rows = [
            row
            for record in records
            for row in record["residency"]["phase_device_memory_stats"].values()
            if row is not None
        ]
        passes.append(
            {
                "surface": value["surface"],
                "declared_scheduled_items": value["declared_scheduled_items"],
                "completed_items": value["completed_items"],
                "checkpoint_count": value["checkpoint_count"],
                "completion_seconds": value["completion_seconds"],
                "host_rss_first_bytes": rss[0] if rss else None,
                "host_rss_peak_bytes": max(rss, default=0),
                "device_limit_bytes": max(
                    (row["bytes_limit"] for row in device_rows), default=0
                ),
                "device_peak_pool_bytes": max(
                    (row["peak_pool_bytes"] for row in device_rows), default=0
                ),
                "device_peak_bytes_in_use": max(
                    (row["peak_bytes_in_use"] for row in device_rows), default=0
                ),
                "compile_events": value["compile_events"],
                "compilation_seconds": value["compilation_seconds"],
                "throughput_source_units_per_second": value[
                    "throughput_source_units_per_second"
                ],
                "prepared_execution_digests": value["prepared_execution_digests"],
                "runtime_callable": value["runtime_callable"],
            }
        )
    payload = {
        "schema_version": "radjax.student.p6_6_reduced_burn_evidence.v1",
        "raw_path": str(args.raw),
        "raw_digest": "sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        "authority": raw["authority"],
        "policy": raw["policy"],
        "platform_preference": raw["platform_preference"],
        "compilation_policy": raw["compilation_policy"],
        "passes": passes,
        "blocked": raw["blocked"],
        "classification": raw["classification"],
        "active_p6_5_findings": [
            "execution_retention_observed",
            "repeated_jit_compilation_observed",
            "owner_unresolved",
        ],
        "limitations": [
            "corridor_envelope_blocked_before_exemplar_pass",
            "no_student_owned_memory_management_change",
            "p6_5_jit_finding_carried_forward_from_pinned_t4_receipt",
        ],
    }
    payload["evidence_identity"] = _digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
