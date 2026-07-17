"""Generate or check the P3.12A executed objective-identity receipt."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from radjax_student.validation.p3_12a_objective_identity.models import build_receipt


def _bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-recorded", action="store_true")
    group.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    # Execution imports JAX only after command parsing.
    from radjax_student.validation.p3_12a_objective_identity.runner_jax import (
        execute_objective_identity_proof,
    )

    with tempfile.TemporaryDirectory(prefix="radjax-p312a-") as temporary:
        payload = build_receipt(execute_objective_identity_proof(Path(temporary)))
    generated = _bytes(payload)
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(generated)
        print(f"P3.12A objective identity receipt written: {args.write}")
        return 0
    recorded = Path("docs/P3_12A_OBJECTIVE_IDENTITY_RECEIPT.json").read_bytes()
    if generated != recorded:
        print("p312a_receipt_mismatch")
        return 1
    print("P3.12A objective identity receipt matches executed proof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
