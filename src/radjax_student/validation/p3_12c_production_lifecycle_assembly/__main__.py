"""Execute, write, or verify the deterministic P3.12C receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import build_receipt


def _bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diagnostic", action="store_true")
    group.add_argument("--check-recorded", action="store_true")
    group.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    from .runner_jax import execute_lifecycle_assembly_proof, raw_diagnostic_failures

    if args.diagnostic:
        failures = raw_diagnostic_failures()
        for case_id in failures:
            print(case_id)
        return 1 if failures else 0
    with TemporaryDirectory(prefix="radjax-p312c-") as temporary:
        generated = _bytes(
            build_receipt(execute_lifecycle_assembly_proof(Path(temporary)))
        )
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(generated)
        return 0
    recorded = Path("docs/P3_12C_PRODUCTION_LIFECYCLE_ASSEMBLY_RECEIPT.json")
    if recorded.read_bytes() != generated:
        print("p312c_receipt_mismatch")
        return 1
    print("P3.12C production lifecycle assembly: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
