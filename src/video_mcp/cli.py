"""Command-line entry point for video-mcp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from video_mcp import __version__
from video_mcp.config import ConfigurationError, load_config
from video_mcp.doctor import format_doctor_report, run_doctor
from video_mcp.errors import VideoMcpError
from video_mcp.logging_config import configure_logging, get_job_logger
from video_mcp.media.ffmpeg import extract_audio
from video_mcp.media.probe import format_media_info, probe_video


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
    doctor_parser = commands.add_parser(
        "doctor",
        help="Report local tool, hardware, model, and workspace capabilities.",
    )
    doctor_parser.add_argument(
        "--json", action="store_true", help="Emit the report as JSON."
    )
    inspect_parser = commands.add_parser(
        "inspect",
        help="Inspect a media file with FFprobe.",
    )
    inspect_parser.add_argument("input", type=Path, help="Input video path.")
    inspect_parser.add_argument(
        "--json", action="store_true", help="Emit normalized metadata as JSON."
    )
    extract_parser = commands.add_parser(
        "extract-audio",
        help="Extract mono 16 kHz PCM WAV audio for transcription.",
    )
    extract_parser.add_argument("input", type=Path, help="Input video path.")
    extract_parser.add_argument(
        "--output", type=Path, help="Output WAV path (defaults inside the workspace)."
    )
    extract_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output WAV."
    )
    extract_parser.add_argument(
        "--json", action="store_true", help="Emit the output paths as JSON."
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

    if args.command == "doctor":
        report = run_doctor(config)
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        else:
            print(format_doctor_report(report))
        return 0 if report.core_ready else 1

    if args.command == "inspect":
        try:
            info = probe_video(args.input, ffprobe_path=config.tools.ffprobe)
        except VideoMcpError as exc:
            print(f"Media error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(info.as_dict(), indent=2))
        else:
            print(format_media_info(info))
        return 0

    if args.command == "extract-audio":
        output = args.output or config.output.workspace / f"{args.input.stem}.wav"
        try:
            audio_path = extract_audio(
                args.input,
                output,
                ffmpeg_path=config.tools.ffmpeg,
                overwrite=args.overwrite,
            )
        except VideoMcpError as exc:
            print(f"Media error: {exc}", file=sys.stderr)
            return 1
        result = {"input": str(args.input.resolve()), "audio": str(audio_path)}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Audio: {audio_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
