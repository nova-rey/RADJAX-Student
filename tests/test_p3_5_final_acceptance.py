from __future__ import annotations

import json
from pathlib import Path

from radjax_student.learning.p3_5_acceptance import (
    FLAGS,
    SCHEMA,
    run_p3_5_architecture_integrity_acceptance,
)


def test_p3_5_final_receipt_is_immutable_and_machine_readable():
    receipt = run_p3_5_architecture_integrity_acceptance()
    recorded = json.loads(
        (
            Path(__file__).parents[1] / "docs/P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json"
        ).read_text()
    )
    assert receipt.schema_version == SCHEMA
    assert receipt.status == "pass"
    assert all(getattr(receipt, name) for name in FLAGS)
    assert recorded == receipt.to_dict()
