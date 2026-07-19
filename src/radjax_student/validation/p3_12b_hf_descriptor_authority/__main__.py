"""Execute or check the generated P3.12B descriptor-authority receipt."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from radjax_student.validation.p3_12b_hf_descriptor_authority.models import (
    build_receipt,
)


def _bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-recorded", action="store_true")
    group.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    from radjax_student.validation.p3_12b_hf_descriptor_authority.runner_jax import (
        execute_hf_descriptor_authority_proof,
    )

    with tempfile.TemporaryDirectory(prefix="radjax-p312b-") as temporary:
        generated = _bytes(
            build_receipt(execute_hf_descriptor_authority_proof(Path(temporary)))
        )
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(generated)
        return 0
    recorded = Path("docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_bytes()
    if recorded != generated:
        print("p312b_receipt_mismatch")
        return 1
    print("P3.12B HF descriptor authority: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
