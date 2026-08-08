"""Command-line entry point for video-mcp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from video_mcp import __version__
from video_mcp.config import ConfigurationError, load_config
from video_mcp.logging_config import configure_logging, get_job_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-mcp",
        description="Local-first video transcription and subtitle tooling.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to video-mcp.yaml (defaults to the current directory).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    commands = parser.add_subparsers(dest="command")
    commands.add_parser(
        "config",
        help="Print the effective configuration after environment overrides.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        config = load_config(args.config)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logger = get_job_logger("video_mcp.cli", command=args.command)
    logger.debug("Command started")

    if args.command == "config":
        print(json.dumps(config.as_dict(), indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
