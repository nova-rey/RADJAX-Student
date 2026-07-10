from __future__ import annotations

import argparse
from typing import TextIO

from radjax_student.cli.render import (
    render_doctor_human,
    render_json,
    write_rendered_output,
)
from radjax_student.reports import build_doctor_report


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="verify the local Phase 1 artifact-understanding pipeline",
    )
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="run the explicit P2.4 JAX CPU execution smoke",
    )
    parser.add_argument("--output", help="write the rendered report to this path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing --output file",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, stdout: TextIO) -> int:
    report = build_doctor_report(run_runtime_smoke=args.runtime_smoke)
    rendered = (
        render_json(report) if args.format == "json" else render_doctor_human(report)
    )
    if args.output:
        output_path = write_rendered_output(
            args.output,
            rendered,
            overwrite=args.overwrite,
        )
        stdout.write(f"Wrote report: {output_path}\n")
    else:
        stdout.write(rendered.rstrip() + "\n")
    return 0 if report.ok else 1
