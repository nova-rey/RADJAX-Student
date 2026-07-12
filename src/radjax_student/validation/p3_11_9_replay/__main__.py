"""Executable P3.11.9 replay gate."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from radjax_student.validation.p3_11_9_replay.artifact import build_replay_receipt
from radjax_student.validation.p3_11_9_replay.documentation import check_documentation
from radjax_student.validation.p3_11_9_replay.verifier import (
    recorded_artifact_difference,
    verify_recorded_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3.11.9 deterministic replay gate")
    parser.add_argument("--check-recorded", action="store_true")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    if args.check_recorded == (args.write is not None):
        parser.error("specify exactly one of --check-recorded or --write")
    repository_root = Path(__file__).resolve().parents[4]
    # The JAX runner is intentionally lazy: passive replay imports remain JAX-free.
    from radjax_student.validation.p3_11_9_replay.runner_jax import (
        execute_stateful_replays,
    )

    with tempfile.TemporaryDirectory(prefix="radjax-p3119-") as temporary:
        proof = execute_stateful_replays(Path(temporary))
    receipt = build_replay_receipt(proof)
    data = receipt.to_json_bytes()
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes(data)
        print(f"P3.11.9 replay evidence written: {args.write}")
        return 0
    recorded_path = repository_root / "docs/P3_11_9_REPLAY_EVIDENCE.json"
    recorded = recorded_path.read_bytes()
    verification = verify_recorded_artifact(data, recorded)
    documentation = check_documentation(repository_root, recorded)
    if not verification.passed:
        print(
            "P3.11.9 replay failed: "
            + ", ".join(item.code for item in verification.blockers)
        )
        print(
            "P3.11.9 replay difference: " + recorded_artifact_difference(data, recorded)
        )
        return 1
    if not documentation.ok:
        print("P3.11.9 documentation failed: " + ", ".join(documentation.errors))
        return 1
    print("P3.11.9 deterministic replay: pass")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
