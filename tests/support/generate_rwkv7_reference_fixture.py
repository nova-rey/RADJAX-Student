"""Generate the checked-in P4.4 RWKV-7 tiny-domain parity fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.support.rwkv7_reference_oracle import (
    fixture_carry,
    fixture_parameters,
    rwkv7_sequence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/rwkv7_reference/parity_fixture.json"
PROVENANCE_PATH = REPOSITORY_ROOT / "tests/fixtures/rwkv7_reference/provenance.json"
PINNED_SOURCE = {
    "repository": "BlinkDL/RWKV-LM",
    "commit": "442120a5b40f7d764328bebde94324bc8790806f",
    "path": "RWKV-v7/rwkv_v7_numpy.py",
    "sha256": "dd683466cf97880c82879afbc8abb27a9596b12344a825d8325a1a1753597ee6",
}
TOKENS = (1, 7, 3, 5)


def _json_value(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {name: _json_value(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def fixture_payload() -> dict[str, object]:
    parameters = fixture_parameters()
    logits, carry = rwkv7_sequence(parameters, TOKENS, fixture_carry())
    return {
        "schema_version": "radjax.rwkv7_reference_parity_fixture.v1",
        "pinned_source": PINNED_SOURCE,
        "domain": {
            "vocabulary_size": 16,
            "hidden_size": 8,
            "layer_count": 2,
            "head_count": 2,
            "head_size": 4,
            "ffn_width": 16,
            "context_length": 4,
            "dtype": "float32",
        },
        "parameter_generator": "deterministic_trigonometric_fixture.v1",
        "tokens": list(TOKENS),
        "expected_logits": _json_value(logits),
        "expected_carry": _json_value(carry),
    }


def fixture_bytes() -> bytes:
    payload = json.dumps(fixture_payload(), sort_keys=True, separators=(",", ":"))
    return (payload + "\n").encode()


def provenance_payload(fixture: bytes | None = None) -> dict[str, object]:
    fixture = fixture_bytes() if fixture is None else fixture
    generator = Path(__file__).read_bytes()
    oracle = (REPOSITORY_ROOT / "tests/support/rwkv7_reference_oracle.py").read_bytes()
    return {
        "schema_version": "radjax.rwkv7_reference_parity_provenance.v1",
        "pinned_source": PINNED_SOURCE,
        "generator": {
            "path": "tests/support/generate_rwkv7_reference_fixture.py",
            "sha256": hashlib.sha256(generator).hexdigest(),
        },
        "oracle": {
            "path": "tests/support/rwkv7_reference_oracle.py",
            "sha256": hashlib.sha256(oracle).hexdigest(),
        },
        "fixture": {
            "path": "tests/fixtures/rwkv7_reference/parity_fixture.json",
            "sha256": hashlib.sha256(fixture).hexdigest(),
        },
    }


def provenance_bytes(fixture: bytes | None = None) -> bytes:
    return (
        json.dumps(provenance_payload(fixture), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


if __name__ == "__main__":
    print(fixture_bytes().decode(), end="")
