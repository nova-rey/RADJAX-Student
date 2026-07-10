from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from radjax_student.artifacts import TomeArtifactError
from radjax_student.cli import doctor, inspect
from radjax_student.cli.render import OutputExistsError

EXIT_SUCCESS = 0
EXIT_COMPATIBILITY_FAIL = 1
EXIT_USAGE_OR_ARTIFACT_ERROR = 2
EXIT_INTERNAL_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radjax-student",
        description="Inspect validated RADJAX artifacts before runtime action.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect.configure_parser(subparsers)
    doctor.configure_parser(subparsers)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = sys.stdout if stdout is None else stdout
    error_output = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.command_handler(args, output))
    except TomeArtifactError as exc:
        error_output.write("Artifact could not be opened.\n\nBlockers:\n")
        for blocker in exc.blockers:
            error_output.write(f"- {blocker}\n")
        return EXIT_USAGE_OR_ARTIFACT_ERROR
    except (OutputExistsError, ValueError) as exc:
        error_output.write(f"Error: {exc}\n")
        return EXIT_USAGE_OR_ARTIFACT_ERROR
    except Exception as exc:  # Normal CLI mode converts unexpected errors to code 3.
        error_output.write(f"Internal error: {type(exc).__name__}: {exc}\n")
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
