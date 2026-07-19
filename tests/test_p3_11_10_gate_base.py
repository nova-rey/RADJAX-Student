"""Passive contract and inventory tests for P3.11.10."""
# ruff: noqa: E501

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
    canonical_json_bytes,
    parse_canonical_json,
)
from radjax_student.validation.p3_11_10_gate.documentation import (
    check_closure_documentation,
)
from radjax_student.validation.p3_11_10_gate.gate import (
    GateInventoryError,
    execute_case,
    validate_inventory,
)
from radjax_student.validation.p3_11_10_gate.inventory import (
    CASES,
    SECTIONS,
    expected_case_ids,
)
from radjax_student.validation.p3_11_10_gate.models import (
    FinalAdversarialGateReceipt,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/P3_11_10_FINAL_ADVERSARIAL_GATE_RECEIPT.json"
EXPECTED_SECTION_COUNTS = {
    "A": 18,
    "B": 19,
    "C": 14,
    "D": 19,
    "E": 21,
    "F": 22,
    "G": 43,
    "H": 20,
    "I": 40,
    "J": 18,
    "K": 7,
}


def _encoded(payload: dict) -> bytes:
    payload["gate_evidence_digest"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "gate_evidence_digest"}
    )
    return canonical_json_bytes(payload)


def test_passive_gate_import_does_not_load_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import radjax_student.validation.p3_11_10_gate; "
            "import radjax_student.validation.p3_11_10_gate.models; "
            "import radjax_student.validation.p3_11_10_gate.inventory; "
            "assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_complete_canonical_inventory_is_ordered_and_exactly_sectioned():
    assert {
        section: len(expected_case_ids(section)) for section in SECTIONS
    } == EXPECTED_SECTION_COUNTS
    assert len(CASES) == sum(EXPECTED_SECTION_COUNTS.values()) == 241
    assert len({case.case_id for case in CASES}) == len(CASES)
    assert all(case.case_id.startswith(f"{case.section_id}.") for case in CASES)
    assert validate_inventory() == CASES


def test_inventory_rejects_missing_duplicate_and_wrong_section_before_dispatch():
    with pytest.raises(GateInventoryError, match="missing_or_ordered"):
        validate_inventory(CASES[1:])
    duplicate = (*CASES, CASES[0])
    with pytest.raises(GateInventoryError):
        validate_inventory(duplicate)
    wrong = list(CASES)
    wrong[0] = type(wrong[0])(
        wrong[0].case_id,
        "B",
        wrong[0].execution_class,
        wrong[0].expected_outcome,
        wrong[0].expected_failure,
        wrong[0].boundary,
        wrong[0].description,
    )
    with pytest.raises(GateInventoryError):
        validate_inventory(tuple(wrong))


def test_base_negative_case_invokes_boundary_twice_and_requires_expected_failure():
    case = next(
        item for item in CASES if item.case_id == "A.reject.architecture_apply_jax_only"
    )
    result = execute_case(case, ROOT)
    assert result.passed
    assert result.observed_outcome == "reject"
    assert result.intended_boundary_reached
    assert result.repeated_first_failure
    assert result.observed_failure is not None
    assert result.observed_failure.code == case.expected_failure
    assert (
        result.observed_failure.details["exception_type"] == "ArchitectureContractError"
    )


def test_recorded_receipt_is_strict_and_evidence_coupled():
    parsed = FinalAdversarialGateReceipt.from_json_bytes(RECEIPT.read_bytes())
    assert parsed["schema_version"] == "radjax.p3_11_10_final_adversarial_gate.v1"
    assert parsed["status"] == "pass"
    assert parsed["ordered_case_ids"] == [case.case_id for case in CASES]
    assert parsed["positive_control_count"] == 15
    assert parsed["adversarial_case_count"] == 226
    assert parsed["unexpected_pass_count"] == 0
    assert parsed["unexpected_failure_count"] == 0


def test_receipt_rejects_omitted_case_and_handwritten_count_preserving_edits():
    payload = copy.deepcopy(parse_canonical_json(RECEIPT.read_bytes()))
    payload["sections"][0]["cases"].pop()
    payload["sections"][0]["ordered_case_ids"].pop()
    payload["sections"][0]["expected_case_count"] -= 1
    payload["sections"][0]["executed_case_count"] -= 1
    payload["ordered_case_ids"].pop(17)
    payload["adversarial_case_count"] -= 1
    with pytest.raises(ReplayCanonicalError):
        FinalAdversarialGateReceipt.from_json_bytes(_encoded(payload))


def test_closure_documentation_allowlist_is_consistent():
    assert check_closure_documentation(ROOT).ok


def test_closure_documentation_binds_final_gate_digest_to_recorded_receipt(
    tmp_path: Path,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    document = (ROOT / "docs/P3_11_10_FINAL_ADVERSARIAL_GATE.md").read_text(
        encoding="utf-8"
    )
    recorded_digest = parse_canonical_json(RECEIPT.read_bytes())["gate_evidence_digest"]
    (docs / "P3_11_10_FINAL_ADVERSARIAL_GATE.md").write_text(
        document.replace(recorded_digest, "0" * 64, 1), encoding="utf-8"
    )
    (docs / "P3_11_10_FINAL_ADVERSARIAL_GATE_RECEIPT.json").write_bytes(
        RECEIPT.read_bytes()
    )
    for relative in (
        "README.md",
        "docs/INDEX.md",
        "docs/ROADMAP.md",
        "docs/RADJAX_DEVELOPMENT_ROADMAP.md",
        "docs/RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md",
        "docs/P3_11_7_CHECKPOINT_V3.md",
        "docs/P3_11_8_STATEFUL_SYSTEMS_PROOF.md",
        "docs/P3_11_9_DETERMINISTIC_REPLAY.md",
        "docs/P3_11_INTEGRATION_CLOSURE.md",
        "docs/P3_5_ARCHITECTURE_INTEGRITY_ROADMAP.md",
        "docs/P3_5_10_FINAL_ARCHITECTURE_INTEGRITY_GATE.md",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    check = check_closure_documentation(tmp_path)
    assert check.errors == ("final_gate_digest_documentation_stale",)
