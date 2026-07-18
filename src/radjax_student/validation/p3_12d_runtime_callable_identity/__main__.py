"""Execute, write, or verify the deterministic P3.12D receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import build_receipt, validate_receipt
from .runner_jax import execute_runtime_callable_identity_proof, raw_diagnostic_failures

_RECEIPT_PATH = Path("docs/P3_12D_RUNTIME_CALLABLE_IDENTITY_RECEIPT.json")


def _bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _receipt(root: Path) -> dict[str, object]:
    return build_receipt(execute_runtime_callable_identity_proof(root))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check-recorded", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    arguments = parser.parse_args()
    if arguments.diagnostic:
        failures = raw_diagnostic_failures()
        for case_id in failures:
            print(case_id)
        return 1 if failures else 0
    if arguments.write:
        with TemporaryDirectory(prefix="radjax-p312d-write-") as temporary:
            payload = _receipt(Path(temporary))
        _RECEIPT_PATH.write_bytes(_bytes(payload))
        return 0
    if arguments.check_recorded:
        if not _RECEIPT_PATH.is_file():
            print("p312d_receipt_missing")
            return 1
        try:
            recorded = validate_receipt(json.loads(_RECEIPT_PATH.read_text()))
        except Exception:
            print("p312d_receipt_invalid")
            return 1
        with (
            TemporaryDirectory(prefix="radjax-p312d-check-") as first,
            TemporaryDirectory(prefix="radjax-p312d-check-") as second,
        ):
            first_bytes = _bytes(_receipt(Path(first)))
            second_bytes = _bytes(_receipt(Path(second)))
        if first_bytes != second_bytes or first_bytes != _bytes(recorded):
            print("p312d_receipt_mismatch")
            return 1
        return 0
    parser.error("one of --write, --check-recorded, or --diagnostic is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
