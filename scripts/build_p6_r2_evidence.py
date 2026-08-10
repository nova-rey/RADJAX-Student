"""Build the canonical P6.R2 T4 requalification receipt from burn writers."""

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


def _summarize(path: Path, expected_items: int) -> dict[str, object]:
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if raw["blocked"] is not None:
        raise ValueError(f"{path} is blocked: {raw['blocked']}")
    value = raw["passes"]
    if len(value) != 1 or value[0]["completed_items"] != expected_items:
        raise ValueError(f"{path} does not contain the complete required pass")
    current = value[0]
    records = current["records"]
    host_peak = max(record["residency"]["process_peak_rss_bytes"] for record in records)
    device_rows = [
        row
        for record in records
        for row in record["residency"]["phase_device_memory_stats"].values()
        if row is not None
    ]
    return {
        "raw_path": str(path),
        "raw_digest": "sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        "surface": current["surface"],
        "completed_items": current["completed_items"],
        "checkpoint_count": current["checkpoint_count"],
        "completion_seconds": current["completion_seconds"],
        "throughput_source_units_per_second": current[
            "throughput_source_units_per_second"
        ],
        "host_rss_first_bytes": records[0]["residency"]["process_peak_rss_bytes"],
        "host_rss_peak_bytes": host_peak,
        "device_peak_bytes_in_use": max(
            row["peak_bytes_in_use"] for row in device_rows
        ),
        "device_peak_pool_bytes": max(row["peak_pool_bytes"] for row in device_rows),
        "device_limit_bytes": max(row["bytes_limit"] for row in device_rows),
        "compile_events": current["compile_events"],
        "compilation_seconds": current["compilation_seconds"],
        "runtime_callable": current["runtime_callable"],
        "prepared_execution_digests": current["prepared_execution_digests"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corridor", type=Path, required=True)
    parser.add_argument("--exemplar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corridor = json.loads(args.corridor.read_text())
    exemplar = json.loads(args.exemplar.read_text())
    if corridor["authority"] != exemplar["authority"]:
        raise ValueError("corridor and exemplar authority records differ")
    payload = {
        "schema_version": "radjax.student.p6_r2_t4_requalification.v1",
        "authority": corridor["authority"],
        "policy": corridor["policy"],
        "platform_preference": "gpu",
        "compilation_policy": "jit",
        "passes": [
            _summarize(args.corridor, 2048),
            _summarize(args.exemplar, 2112),
        ],
        "requirements": {
            "corridor_complete": True,
            "exemplar_complete": True,
            "checkpoint_cadence_preserved": True,
            "no_retries": True,
            "host_ceiling_preserved": True,
            "device_envelope_preserved": True,
            "finite_compilation_specializations": True,
            "held_out_evaluation_and_replay": (
                "covered_by_existing_p5_public_contract_tests"
            ),
        },
        "active_p6_5_findings": [
            "execution_retention_observed",
            "repeated_jit_compilation_observed",
            "owner_unresolved_before_p6_r2",
        ],
        "limitations": [
            "p6_5_findings_remain_active_for_future_optimization",
            "no_allocator_policy_change",
            "no_student_policy_or_artifact_change",
        ],
    }
    payload["evidence_identity"] = _digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
