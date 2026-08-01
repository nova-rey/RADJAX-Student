from __future__ import annotations

import pytest
from radjax_contract.tome import tome_student_consumption_v4_contract_root

from radjax_student.artifacts import (
    NATIVE_V3_STUDENT_PROFILE,
    NativeV3StudentConsumptionError,
    load_native_v3_contract_assets,
    open_native_v3_student_consumption,
)


def test_native_v3_contract_assets_are_discovered_and_checksum_closed() -> None:
    assets = load_native_v3_contract_assets()

    assert assets.root == tome_student_consumption_v4_contract_root()
    assert assets.contract_id == "radjax_tome_student_consumption_contract"
    assert assets.publication_version == "4.0.0"
    assert assets.profile_id == NATIVE_V3_STUDENT_PROFILE
    assert "contract.json" in assets.asset_digests
    assert "fixtures/catalog.json" in assets.asset_digests
    assert "fixtures/valid/native_v3_student_v4.json" in assets.asset_digests
    assert "fixtures/adversarial/cases.json" in assets.asset_digests
    assert len(assets.asset_digests) == 15


def test_native_v3_admission_preserves_contract_issue_codes(tmp_path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(NativeV3StudentConsumptionError) as exc_info:
        open_native_v3_student_consumption(missing)

    assert exc_info.value.path == missing
    assert exc_info.value.issue_codes == ("TSC020_TRANSPORT_UNSUPPORTED",)
