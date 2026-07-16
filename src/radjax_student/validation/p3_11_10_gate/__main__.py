"""Executable P3.11.10 final adversarial integration gate."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from radjax_student.validation.p3_11_10_gate.gate import build_receipt, execute_gate
from radjax_student.validation.p3_11_10_gate.implementations import (
    CASE_IMPLEMENTATIONS,
)
from radjax_student.validation.p3_11_10_gate.models import FinalAdversarialGateReceipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3.11.10 final adversarial gate")
    parser.add_argument("--check-recorded", action="store_true")
    parser.add_argument("--write", type=Path)
    parser.add_argument("--print-implementation-audit", action="store_true")
    args = parser.parse_args(argv)
    selected = sum(
        (
            args.check_recorded,
            args.write is not None,
            args.print_implementation_audit,
        )
    )
    if selected != 1:
        parser.error(
            "specify exactly one of --check-recorded, --write, or "
            "--print-implementation-audit"
        )
    root = Path(__file__).resolve().parents[4]
    proof = execute_gate(root)
    receipt = build_receipt(proof)
    data = receipt.to_json_bytes()
    if args.print_implementation_audit:
        print("Case\tFunction\tPublic Callable\tMutation")
        for record in receipt.to_dict()["implementation_audit"]:
            implementation = CASE_IMPLEMENTATIONS[record["case_id"]]
            print(
                "\t".join(
                    (
                        record["case_id"],
                        implementation.function.__qualname__,
                        record["public_callable_identity"],
                        record["mutation_operation"],
                    )
                )
            )
        return 0
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(data)
        print(f"P3.11.10 final gate receipt written: {args.write}")
        return 0
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "radjax_student.validation.p3_11_9_replay",
            "--check-recorded",
        ],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    if replay.returncode:
        print("P3.11.10 replay prerequisite failed: " + replay.stdout + replay.stderr)
        return 1
    recorded_path = root / "docs/P3_11_10_FINAL_ADVERSARIAL_GATE_RECEIPT.json"
    if not recorded_path.is_file():
        print("P3.11.10 recorded receipt is missing")
        return 1
    recorded = recorded_path.read_bytes()
    try:
        FinalAdversarialGateReceipt.from_json_bytes(recorded)
    except Exception as error:
        print(f"P3.11.10 recorded receipt invalid: {error}")
        return 1
    if data != recorded:
        print("P3.11.10 receipt mismatch")
        return 1
    print("P3.11.10 final adversarial gate: pass")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
