"""Passive strict-schema contracts for P3.11.9 replay evidence."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
    canonical_json_bytes,
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


def _artifact_payload() -> dict:
    return copy.deepcopy(parse_canonical_json(ARTIFACT.read_bytes()))


def _encoded(payload: dict) -> bytes:
    payload["evidence_digest"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "evidence_digest"}
    )
    return canonical_json_bytes(payload)


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
    payload = _artifact_payload()
    payload["unknown"] = True
    with pytest.raises(ReplayCanonicalError, match="fields differ"):
        StatefulReplayReceipt.from_json_bytes(_encoded(payload))


@pytest.mark.parametrize(
    "path",
    [
        ("experiment_identity",),
        ("experiment_identity", "hf_reference"),
        ("experiment_identity", "architecture_carry_descriptor"),
        ("modes", "eager", "canonical_trace", "steps", 0, "runtime"),
        ("modes", "eager", "canonical_trace", "steps", 0, "rng"),
        ("cross_mode_tolerance",),
        ("cross_mode",),
        ("verifier",),
    ],
)
def test_each_nested_contract_rejects_unknown_and_missing_fields(path):
    for operation in ("unknown", "missing"):
        payload = _artifact_payload()
        target = payload
        for part in path:
            target = target[part]
        key = next(iter(target))
        if operation == "unknown":
            target["unexpected"] = True
        else:
            target.pop(key)
        with pytest.raises(ReplayCanonicalError, match="fields differ"):
            StatefulReplayReceipt.from_json_bytes(_encoded(payload))


@pytest.mark.parametrize(
    "path,value",
    [
        (("experiment_identity", "root_seed"), "17"),
        (("experiment_identity", "parameter_catalog_digest"), "A" * 64),
        (("experiment_identity", "hf_reference", "vocabulary_size"), "8"),
        (
            (
                "experiment_identity",
                "architecture_carry_descriptor",
                "pytree_descriptor_digest",
            ),
            "0",
        ),
        (
            ("modes", "eager", "canonical_trace", "steps", 0, "runtime", "compiled"),
            "false",
        ),
        (("modes", "eager", "canonical_trace", "steps", 0, "rng", "global_step"), -1),
        (("cross_mode_tolerance", "rtol"), "0x1p-20"),
        (("cross_mode_tolerance", "atol"), "nan"),
        (("cross_mode", "declared_rtol"), "0x1p-20"),
        (("verifier", "status"), "maybe"),
    ],
)
def test_nested_contracts_reject_malformed_values(path, value):
    payload = _artifact_payload()
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ReplayCanonicalError):
        StatefulReplayReceipt.from_json_bytes(_encoded(payload))


def test_unknown_nested_runtime_field_fails_after_digest_recomputation():
    payload = _artifact_payload()
    payload["modes"]["eager"]["canonical_trace"]["steps"][0]["runtime"]["timing"] = (
        "forbidden"
    )
    with pytest.raises(ReplayCanonicalError, match="runtime fields differ"):
        StatefulReplayReceipt.from_json_bytes(_encoded(payload))


def test_duplicate_nested_json_field_is_rejected_before_schema_validation():
    text = ARTIFACT.read_text()
    duplicate = text.replace(
        '"backend_id":"jax",',
        '"backend_id":"jax","backend_id":"jax",',
        1,
    )
    with pytest.raises(ReplayCanonicalError, match="duplicate"):
        StatefulReplayReceipt.from_json_bytes(duplicate)


def test_recorded_artifact_and_documentation_are_strictly_consistent():
    artifact = ARTIFACT.read_bytes()
    parsed = StatefulReplayReceipt.from_json_bytes(artifact)
    assert parsed["schema_version"] == "radjax.p3_11_9_replay_evidence.v1"
    assert parsed["status"] == "pass"
    assert check_documentation(ROOT, artifact).ok
