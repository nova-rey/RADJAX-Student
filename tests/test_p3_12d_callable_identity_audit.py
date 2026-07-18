"""Base-suite coverage for the P3.12D source audit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from radjax_student.validation.p3_12d_runtime_callable_identity import (
    implementation_audit,
)
from radjax_student.validation.p3_12d_runtime_callable_identity.audit_fixtures import (
    REQUIRED_BAD_SOURCE_FIXTURES,
)
from radjax_student.validation.p3_12d_runtime_callable_identity.inventory import (
    ADVERSARIAL_CASE_IDS,
    POSITIVE_CASE_IDS,
)


def test_inventory_and_real_audit_are_jax_free() -> None:
    assert len(POSITIVE_CASE_IDS) == 18
    assert len(ADVERSARIAL_CASE_IDS) == 40
    assert implementation_audit.audit_runtime_callable_identity().status == "pass"
    script = (
        "import sys; "
        "from radjax_student.validation.p3_12d_runtime_callable_identity "
        "import implementation_audit; "
        "implementation_audit.audit_runtime_callable_identity(); "
        "assert not any(name == 'jax' or name.startswith('jax.') "
        "for name in sys.modules)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd() / "src"),
        },
    )
    assert result.returncode == 0, result.stderr


def test_all_required_synthetic_bad_sources_execute_the_audit() -> None:
    assert len(REQUIRED_BAD_SOURCE_FIXTURES) == 28
    for fixture in REQUIRED_BAD_SOURCE_FIXTURES:
        blockers = implementation_audit.audit_synthetic_source(fixture.source)
        assert [item.code for item in blockers] == [fixture.expected_blocker]
