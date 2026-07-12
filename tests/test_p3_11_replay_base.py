"""Base-environment contracts for P3.11.9 replay evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_metric_mapping,
    finite_float_hex,
    mapping_pytree_digest,
    parse_canonical_json,
)
from radjax_student.validation.p3_11_9_replay.documentation import (
    check_documentation,
)
from radjax_student.validation.p3_11_9_replay.models import StatefulReplayReceipt

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/P3_11_9_REPLAY_EVIDENCE.json"


def test_passive_replay_imports_do_not_load_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import radjax_student.validation.p3_11_9_replay; "
            "import radjax_student.validation.p3_11_9_replay.canonical; "
            "import radjax_student.validation.p3_11_9_replay.verifier; "
            "import radjax_student.validation.p3_11_9_replay.documentation; "
            "assert not any(name == 'jax' or name.startswith('jax.') "
            "for name in sys.modules)",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_canonical_scalars_and_mapping_pytree_identity_are_stable():
    assert finite_float_hex(0.25) == (0.25).hex()
    assert canonical_metric_mapping({"z": 1.0, "a": np.float32(2.0)}) == {
        "a": float(np.float32(2.0)).hex(),
        "z": (1.0).hex(),
    }
    c_order = {"dense": {"kernel": np.array([[1.0, 2.0]], dtype="<f4", order="C")}}
    f_order = {"dense": {"kernel": np.asfortranarray(c_order["dense"]["kernel"])}}
    assert mapping_pytree_digest(c_order) == mapping_pytree_digest(f_order)
    changed = {"dense": {"kernel": np.array([[1.0, 3.0]], dtype="<f4")}}
    assert mapping_pytree_digest(c_order) != mapping_pytree_digest(changed)
    with pytest.raises(TypeError):
        mapping_pytree_digest({"bad": np.asarray([object()])})


def test_duplicate_unknown_and_nonfinite_evidence_is_rejected():
    with pytest.raises(ReplayCanonicalError, match="duplicate"):
        parse_canonical_json('{"x": 1, "x": 2}')
    with pytest.raises(ReplayCanonicalError):
        finite_float_hex(float("nan"))
    payload = json.loads(ARTIFACT.read_text())
    payload["unknown"] = True
    with pytest.raises(ReplayCanonicalError, match="fields differ"):
        StatefulReplayReceipt.from_json_bytes(json.dumps(payload))


def test_recorded_artifact_and_documentation_are_strictly_consistent():
    artifact = ARTIFACT.read_bytes()
    parsed = StatefulReplayReceipt.from_json_bytes(artifact)
    assert parsed["schema_version"] == "radjax.p3_11_9_replay_evidence.v1"
    assert parsed["status"] == "pass"
    assert check_documentation(ROOT, artifact).ok
