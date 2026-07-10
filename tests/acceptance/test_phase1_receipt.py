from __future__ import annotations

import hashlib
import json

from .support import ACCEPTED_FIXTURE_DIGEST, REPO_ROOT

EXPECTED_NON_CLAIMS = {
    "corridor_payload_loading_not_tested",
    "exemplar_payload_loading_not_tested",
    "loss_computation_not_tested",
    "model_allocation_not_tested",
    "architecture_plugin_support_not_available",
    "runtime_selection_not_available",
    "jax_xla_portability_not_tested",
    "checkpoint_execution_not_tested",
    "schedule_execution_not_tested",
    "training_not_run",
    "evaluation_not_run",
    "hf_export_not_available",
    "functional_stage_distillation_not_available",
    "model_quality_not_claimed",
    "performance_not_claimed",
    "scale_not_claimed",
    "radlads_parity_not_measured",
}


def test_machine_readable_receipt_is_complete_and_self_consistent() -> None:
    receipt = json.loads(
        (REPO_ROOT / "phase1_acceptance_receipt.json").read_text(encoding="utf-8")
    )

    assert receipt["gate"] == "P1.10"
    assert receipt["gate_version"] == 1
    assert receipt["status"] == "pass"
    assert receipt["blockers"] == []
    assert receipt["fixture"]["tree_digest"] == ACCEPTED_FIXTURE_DIGEST
    assert receipt["student"]["accepted_input_commit"].startswith("eff271d")
    assert receipt["contract"]["acceptance_receipt_commit"].startswith("ff8f6e9")
    assert receipt["tome"]["producer_commit"].startswith("fe5d51e")
    assert set(receipt["claims_not_made"]) == EXPECTED_NON_CLAIMS
    assert receipt["phase_status"] == {
        "phase_1_contract_layer": "complete",
        "phase_2_student_runtime": "unblocked",
    }
    assert receipt["test_results"]
    assert receipt["warnings"]

    for golden in receipt["golden_reports"]:
        path = REPO_ROOT / golden["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == golden["sha256"]


def test_human_report_and_roadmap_close_phase1() -> None:
    report = (REPO_ROOT / "docs/P1_10_PHASE1_ACCEPTANCE_GATE.md").read_text(
        encoding="utf-8"
    )
    roadmap = (REPO_ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")

    assert "**Status:** PASS" in report
    assert "PHASE 1 - CONTRACT LAYER" in roadmap
    assert "PHASE 2 - STUDENT RUNTIME" in roadmap
    assert "COMPLETE" in roadmap
    assert "UNBLOCKED" in roadmap
