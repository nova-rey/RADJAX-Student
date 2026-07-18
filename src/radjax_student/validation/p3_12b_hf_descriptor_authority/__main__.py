"""Execute or check the generated P3.12B descriptor-authority receipt."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from radjax_student.validation.p3_12b_hf_descriptor_authority.models import (
    build_receipt,
    validate_receipt,
)


def _bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-recorded", action="store_true")
    group.add_argument("--write", type=Path)
    parser.add_argument(
        "--recorded",
        type=Path,
        default=Path("docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json"),
    )
    args = parser.parse_args(argv)
    from radjax_student.validation.p3_12b_hf_descriptor_authority.runner_jax import (
        execute_hf_descriptor_authority_proof,
    )

    with tempfile.TemporaryDirectory(prefix="radjax-p312b-") as temporary:
        proof = execute_hf_descriptor_authority_proof(Path(temporary))
        generated_payload = build_receipt(proof)
        generated = _bytes(generated_payload)
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(generated)
        return 0
    recorded = args.recorded.read_bytes()
    try:
        validate_receipt(json.loads(recorded), proof=proof)
    except (json.JSONDecodeError, ValueError) as error:
        print(f"p312b_receipt_invalid:{error}")
        return 1
    if recorded != generated:
        print("p312b_receipt_mismatch")
        return 1
    print("P3.12B HF descriptor authority: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
