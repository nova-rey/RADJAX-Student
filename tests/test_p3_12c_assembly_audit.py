"""JAX-free P3.12C one-authority source audit tests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.validation.p3_12c_production_lifecycle_assembly import (
    audit_fixtures,
    implementation_audit,
)
from radjax_student.validation.p3_12c_production_lifecycle_assembly.inventory import (
    ADVERSARIAL_CASE_IDS,
    POSITIVE_CASE_IDS,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _audit(
    *,
    source: str = "source",
    blockers: tuple[implementation_audit.AssemblyAuditBlocker, ...] = (),
) -> implementation_audit.AssemblyAuthorityAudit:
    entry = implementation_audit.AssemblyAuditSourceEntry(
        "src/example.py", _sha(source)
    )
    return implementation_audit.AssemblyAuthorityAudit(
        _sha("evidence:" + source),
        (entry,),
        POSITIVE_CASE_IDS,
        ADVERSARIAL_CASE_IDS,
        blockers,
    )


def test_inventory_and_real_source_audit_are_jax_free():
    assert len(POSITIVE_CASE_IDS) == 17
    assert len(ADVERSARIAL_CASE_IDS) == 36
    assert implementation_audit.audit_assembly_authority(Path.cwd()).status == "pass"
    script = (
        "import sys; "
        "from radjax_student.validation.p3_12c_production_lifecycle_assembly "
        "import implementation_audit; "
        "implementation_audit.audit_assembly_authority(); "
        "assert not any(name == 'jax' or name.startswith('jax.') "
        "for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_all_required_synthetic_bad_sources_execute_the_real_audit():
    assert len(audit_fixtures.REQUIRED_BAD_SOURCE_FIXTURES) == 24
    for fixture in audit_fixtures.REQUIRED_BAD_SOURCE_FIXTURES:
        blockers = implementation_audit.audit_synthetic_source(
            fixture.source, path=fixture.path
        )
        assert [item.code for item in blockers] == [fixture.expected_blocker]


def test_typed_audit_round_trip_is_strict_and_status_is_derived():
    passing = _audit()
    assert passing.status == "pass"
    assert (
        implementation_audit.AssemblyAuthorityAudit.from_dict(passing.to_dict())
        == passing
    )
    blocked = _audit(
        blockers=(implementation_audit.AssemblyAuditBlocker("fixture", "detail"),)
    )
    assert blocked.status == "blocked"
    payload = blocked.to_dict()
    payload["status"] = "pass"
    with pytest.raises(ValueError, match="status"):
        implementation_audit.AssemblyAuthorityAudit.from_dict(payload)
    payload = passing.to_dict() | {"unknown": True}
    with pytest.raises(ValueError, match="unknown"):
        implementation_audit.AssemblyAuthorityAudit.from_dict(payload)
    with pytest.raises(ValueError, match="blocker"):
        implementation_audit.AssemblyAuditBlocker.from_dict({"code": "only"})


def test_implementation_digest_is_deterministic_and_evidence_sensitive():
    first = _audit(source="one")
    same = _audit(source="one")
    changed_source = _audit(source="two")
    changed_blocker = _audit(
        source="one",
        blockers=(implementation_audit.AssemblyAuditBlocker("fixture", "changed"),),
    )
    assert first.implementation_audit_digest == same.implementation_audit_digest
    assert (
        first.implementation_audit_digest != changed_source.implementation_audit_digest
    )
    assert (
        first.implementation_audit_digest != changed_blocker.implementation_audit_digest
    )
