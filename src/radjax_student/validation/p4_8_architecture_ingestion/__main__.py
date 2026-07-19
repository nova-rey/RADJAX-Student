"""Write the deterministic P4.8 report without importing JAX at parse time."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args(argv)
    from radjax_student.validation.p4_8_architecture_ingestion.runner_jax import (
        write_phase4_report,
    )

    write_phase4_report(Path.cwd(), args.workdir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
