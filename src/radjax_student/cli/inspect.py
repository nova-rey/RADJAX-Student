from __future__ import annotations

import argparse
from typing import TextIO

from radjax_student.cli.render import (
    render_inspection_human,
    render_json,
    write_rendered_output,
)
from radjax_student.reports import build_inspection_report
from radjax_student.validation import resolve_profile


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="inspect a production Tome and evaluate declared compatibility",
    )
    parser.add_argument("--tome", required=True, help="path to the Tome artifact")
    parser.add_argument(
        "--profile",
        default="metadata_inspection_only",
        metavar="PROFILE_ID",
        help="compatibility profile (default: metadata_inspection_only)",
    )
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument("--output", help="write the rendered report to this path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing --output file",
    )
    parser.add_argument(
        "--show-contents",
        action="store_true",
        help="include the validated content index in human output",
    )
    parser.set_defaults(command_handler=run)


def run(args: argparse.Namespace, stdout: TextIO) -> int:
    profile = resolve_profile(args.profile)
    report = build_inspection_report(args.tome, profile)
    rendered = (
        render_json(report)
        if args.format == "json"
        else render_inspection_human(report, show_contents=args.show_contents)
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
    return 0 if report.status == "pass" else 1
