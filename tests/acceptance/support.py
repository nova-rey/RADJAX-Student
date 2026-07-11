from __future__ import annotations

import hashlib
import json
import shutil
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from radjax_contract.testing import production_tome_fixture_path

from radjax_student.cli.main import main
from radjax_student.reports import (
    StudentDoctorReport,
    StudentInspectionReport,
    build_doctor_report,
    build_inspection_report,
)
from radjax_student.validation import (
    declaration_test_only_profile,
    metadata_inspection_only_profile,
)

ACCEPTED_FIXTURE_DIGEST = (
    "468a259d518a28a6f60af8c339b124b65fd52da0640544d186eb9609933608d1"
)
FIXTURE_ID = "production_multi_surface_v1"
REQUIRED_CAPABILITIES = (
    "radjax.corridor.packed_assignments.v1",
    "radjax.corridor.stat_bands.v1",
    "radjax.exemplar.selected_dynamic_topk.v1",
)
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_fixture() -> Path:
    return production_tome_fixture_path()


def copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "artifact"
    shutil.copytree(canonical_fixture(), destination)
    return destination


def run_cli(*arguments: str) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = main(arguments, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def normalized_inspection_payload() -> dict[str, Any]:
    report = build_inspection_report(
        canonical_fixture(),
        metadata_inspection_only_profile(),
    )
    return normalize_inspection(report)


def normalized_passing_inspection_payload() -> dict[str, Any]:
    report = build_inspection_report(
        canonical_fixture(),
        declaration_test_only_profile(),
    )
    return normalize_inspection(report)


def normalize_inspection(report: StudentInspectionReport) -> dict[str, Any]:
    payload = report.to_dict()
    payload["artifact_path"] = f"<CONTRACT_FIXTURE:{FIXTURE_ID}>"
    return payload


def normalized_doctor_payload(report: StudentDoctorReport | None = None) -> dict:
    payload = (build_doctor_report() if report is None else report).to_dict()
    payload.pop("runtime_inspection")
    payload.pop("runtime_backend_descriptors")
    payload.pop("runtime_selection")
    payload.pop("runtime_smoke")
    payload.pop("runtime_state_smoke")
    payload.pop("runtime_portability_smoke")
    payload.pop("placement_intent")
    payload.pop("execution_boundary")
    payload["capability_state"].pop("runtime_inspection")
    payload["capability_state"].pop("runtime_backend_registry")
    payload["capability_state"].pop("runtime_backend_selection")
    payload["capability_state"].pop("runtime_cpu_smoke")
    payload["capability_state"].pop("placement_intent")
    payload["capability_state"].pop("execution_boundary")
    payload["capability_state"].pop("runtime_state")
    payload["capability_state"].pop("runtime_portability")
    payload["capability_state"].pop("jax_execution")
    payload["python_version"] = "<PYTHON_VERSION>"
    for package in ("student_package", "contract_package"):
        payload[package]["version"] = "<DISCOVERED_OR_NONE>"
        payload[package]["commit"] = "<DISCOVERED_OR_NONE>"
    return payload


def read_golden(name: str) -> dict:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def mutate_artifact(artifact: Path, mutation: str) -> str:
    cover_path = artifact / "cover_page.json"
    cover = read_json(cover_path)
    blocker = ""
    if mutation == "path_traversal":
        content_ref(cover, "corridor_summary")["path"] = "../outside.json"
        blocker = "content_path_unsafe"
    elif mutation == "absolute_path":
        content_ref(cover, "corridor_summary")["path"] = "/tmp/outside.json"
        blocker = "content_path_unsafe"
    elif mutation == "stale_hash":
        content_ref(cover, "corridor_summary")["sha256"] = "0" * 64
        blocker = "content_hash_mismatch"
    elif mutation == "stale_size":
        content_ref(cover, "corridor_summary")["size_bytes"] += 1
        blocker = "content_size_mismatch"
    elif mutation == "missing_required_role":
        cover["contents"] = [
            item for item in cover["contents"] if item["role"] != "corridor_summary"
        ]
        blocker = "surface_required_role_missing"
    elif mutation == "duplicate_path":
        content_ref(cover, "corridor_summary")["path"] = content_ref(
            cover,
            "corridor_mode_table",
        )["path"]
        blocker = "content_path_duplicate"
    elif mutation == "duplicate_role":
        cover["contents"].append(dict(content_ref(cover, "corridor_summary")))
        blocker = "content_role_cardinality_invalid"
    elif mutation == "invalid_surface_reference":
        cover["behavioral_surfaces"][0]["required_content_roles"].append(
            "future_missing_role"
        )
        blocker = "surface_required_role_missing"
    elif mutation == "invalid_pass_reference":
        cover["recommended_training_plan"]["passes"][0]["surface_id"] = "missing"
        blocker = "training_pass_surface_missing"
    elif mutation == "bad_mode_linkage":
        mode_ref = content_ref(cover, "corridor_assignment_mode_id")
        mode_path = artifact / mode_ref["path"]
        modes = np.load(mode_path, allow_pickle=False)
        modes[0] = 999
        np.save(mode_path, modes, allow_pickle=False)
        refresh_ref(mode_ref, mode_path)
        blocker = "corridor_mode_id_domain_invalid"
    elif mutation == "fingerprint_mode_confusion":
        mode_ref = content_ref(cover, "corridor_assignment_mode_id")
        fingerprint_ref = content_ref(
            cover,
            "corridor_assignment_fingerprint_index",
        )
        mode_ref.update(
            path=fingerprint_ref["path"],
            sha256=fingerprint_ref["sha256"],
            size_bytes=fingerprint_ref["size_bytes"],
        )
        blocker = "corridor_mode_id_domain_invalid"
    elif mutation == "packed_array_length":
        weight_ref = content_ref(cover, "corridor_assignment_weight")
        weight_path = artifact / weight_ref["path"]
        weights = np.load(weight_path, allow_pickle=False)
        np.save(weight_path, weights[:-1], allow_pickle=False)
        refresh_ref(weight_ref, weight_path)
        manifest_ref = content_ref(cover, "corridor_assignment_manifest")
        manifest_path = artifact / manifest_ref["path"]
        manifest = read_json(manifest_path)
        manifest["arrays"]["weight"]["shape"] = [len(weights) - 1]
        write_json(manifest_path, manifest)
        refresh_ref(manifest_ref, manifest_path)
        blocker = "corridor_assignment_shape_invalid"
    elif mutation in {
        "invalid_effective_top_k",
        "selection_mask_mismatch",
        "invalid_token_id",
        "bad_exemplar_corridor_linkage",
    }:
        payload_ref = content_ref(cover, "selected_exemplar_payload_shard")
        payload_path = artifact / payload_ref["path"]
        shard = read_json(payload_path)
        payload = shard["selected_exemplars"][0]
        if mutation == "invalid_effective_top_k":
            payload["dynamic_top_k"]["effective_top_k"] = 5
            blocker = "exemplar_dynamic_top_k_mismatch"
        elif mutation == "selection_mask_mismatch":
            payload["top_selection_mask"][0] = False
            blocker = "exemplar_selection_mask_invalid"
        elif mutation == "invalid_token_id":
            payload["top_token_ids"][0] = payload["vocab_size"]
            blocker = "exemplar_token_id_out_of_range"
        else:
            payload["corridor_mode_id"] = 999
            blocker = "exemplar_corridor_mode_unknown"
        write_json(payload_path, shard)
        refresh_ref(payload_ref, payload_path)
    else:
        raise ValueError(f"unknown acceptance mutation: {mutation}")
    write_json(cover_path, cover)
    return blocker


def add_unknown_optional_content(artifact: Path) -> None:
    future_path = artifact / "future-optional.json"
    write_json(future_path, {"future": True})
    cover = read_json(artifact / "cover_page.json")
    cover["contents"].append(
        {
            "classification": "diagnostic",
            "path": "future-optional.json",
            "required": False,
            "role": "future_optional_diagnostic",
            "sha256": sha256(future_path),
            "size_bytes": future_path.stat().st_size,
        }
    )
    cover["behavioral_surfaces"].append(
        {
            "optional_content_roles": ["future_optional_diagnostic"],
            "prerequisites": [],
            "required_capabilities": [],
            "required_content_roles": [],
            "schema_version": "future_surface_v1",
            "semantics": {"future_semantic": "preserved"},
            "surface_id": "future_optional",
            "surface_kind": "future_optional_kind",
            "target_scope": {"kind": "plugin_defined", "plugin": "future"},
        }
    )
    write_json(artifact / "cover_page.json", cover)


def add_unknown_required_capability(artifact: Path) -> str:
    capability = "radjax.future.required.v1"
    cover = read_json(artifact / "cover_page.json")
    cover["behavioral_surfaces"][0]["required_capabilities"].append(capability)
    cover["recommended_training_plan"]["passes"][0]["required_capabilities"].append(
        capability
    )
    write_json(artifact / "cover_page.json", cover)
    return capability


def rename_indexed_content(artifact: Path) -> None:
    cover = read_json(artifact / "cover_page.json")
    ref = content_ref(cover, "corridor_summary")
    current = artifact / ref["path"]
    renamed = artifact / "renamed" / "summary.data"
    renamed.parent.mkdir()
    current.rename(renamed)
    ref["path"] = "renamed/summary.data"
    write_json(artifact / "cover_page.json", cover)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def content_ref(cover: dict, role: str) -> dict:
    return next(item for item in cover["contents"] if item["role"] == role)


def refresh_ref(ref: dict, path: Path) -> None:
    ref["sha256"] = sha256(path)
    ref["size_bytes"] = path.stat().st_size


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finding_codes(findings: tuple) -> list[str]:
    return [finding.code for finding in findings]
