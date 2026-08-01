"""Student-owned admission for the explicit native-v3 consumption profile.

This module owns the Student boundary, not Tome semantics.  Contract owns the
profile, schemas, semantic resolution, and verified resource handles.  P5.1
therefore exposes only immutable admission metadata; P5.2 is responsible for
turning verified resources into Student payload views.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radjax_contract.tome import (
    TOME_STUDENT_CONSUMPTION_CONTRACT_ID,
    TOME_STUDENT_CONSUMPTION_V4_CONTRACT_PUBLICATION_VERSION,
    StudentConsumptionV4Descriptor,
    tome_student_consumption_v4_contract_root,
    validate_and_resolve_student_consumption_v4,
)

NATIVE_V3_STUDENT_PROFILE = "native_v3_student_v4"


@dataclass(frozen=True)
class NativeV3ContractAssets:
    """Verified installed Contract assets for the explicitly negotiated profile."""

    root: Path
    contract_id: str
    publication_version: str
    profile_id: str
    asset_digests: Mapping[str, str]


@dataclass(frozen=True)
class NativeV3StudentConsumptionView:
    """A Contract-admitted native-v3 artifact, without payload materialization."""

    artifact_path: Path
    contract_assets: NativeV3ContractAssets
    descriptor: StudentConsumptionV4Descriptor


class NativeV3StudentConsumptionError(ValueError):
    """Expose Contract's deterministic issue codes at the Student boundary."""

    def __init__(self, path: str | Path, issue_codes: tuple[str, ...]) -> None:
        self.path = Path(path)
        self.issue_codes = issue_codes
        super().__init__(
            f"could not admit native-v3 Student consumption artifact at {self.path}: "
            + ", ".join(issue_codes)
        )


def load_native_v3_contract_assets() -> NativeV3ContractAssets:
    """Discover and checksum-verify the installed v4 Contract asset tree."""

    root = tome_student_consumption_v4_contract_root()
    expected = _read_checksums(root / "SHA256SUMS")
    observed = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if observed != expected:
        raise ValueError(
            "installed native-v3 Student-consumption Contract checksum mismatch"
        )

    contract = _read_json(root / "contract.json")
    if (
        contract.get("contract_id") != TOME_STUDENT_CONSUMPTION_CONTRACT_ID
        or contract.get("publication_version")
        != TOME_STUDENT_CONSUMPTION_V4_CONTRACT_PUBLICATION_VERSION
        or contract.get("profile_id") != NATIVE_V3_STUDENT_PROFILE
    ):
        raise ValueError(
            "installed native-v3 Student-consumption Contract identity mismatch"
        )
    return NativeV3ContractAssets(
        root=root,
        contract_id=TOME_STUDENT_CONSUMPTION_CONTRACT_ID,
        publication_version=TOME_STUDENT_CONSUMPTION_V4_CONTRACT_PUBLICATION_VERSION,
        profile_id=NATIVE_V3_STUDENT_PROFILE,
        asset_digests=MappingProxyType(dict(sorted(expected.items()))),
    )


def open_native_v3_student_consumption(
    path: str | Path, *, strict: bool = False
) -> NativeV3StudentConsumptionView:
    """Admit only Contract's exact v4 profile; never infer or downgrade one."""

    assets = load_native_v3_contract_assets()
    result = validate_and_resolve_student_consumption_v4(path, strict=strict)
    if not result.ok or result.descriptor is None:
        raise NativeV3StudentConsumptionError(
            path, tuple(issue.code for issue in result.issues)
        )
    return NativeV3StudentConsumptionView(
        artifact_path=Path(path),
        contract_assets=assets,
        descriptor=result.descriptor,
    )


def _read_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in entries
        ):
            raise ValueError(
                "installed native-v3 Student-consumption checksum format invalid"
            )
        entries[relative] = digest
    if not entries:
        raise ValueError(
            "installed native-v3 Student-consumption checksum list is empty"
        )
    return entries


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"installed Contract asset must be a JSON object: {path.name}")
    return MappingProxyType(value)
