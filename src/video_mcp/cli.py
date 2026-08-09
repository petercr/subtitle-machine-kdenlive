"""Command-line entry point for video-mcp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from video_mcp import __version__
from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr.whisper_cpp import WhisperCppBackend
from video_mcp.config import ConfigurationError, load_config
from video_mcp.doctor import format_doctor_report, run_doctor
from video_mcp.errors import VideoMcpError
from video_mcp.logging_config import configure_logging, get_job_logger
from video_mcp.media.ffmpeg import extract_audio
from video_mcp.media.probe import format_media_info, probe_video
from video_mcp.media.render import create_preview
from video_mcp.models import Transcript
from video_mcp.services.captioning import CaptionOptions, caption_video
from video_mcp.subtitles.ass import write_ass
from video_mcp.subtitles.srt import write_srt


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
    transcribe_parser = commands.add_parser(
        "transcribe",
        help="Transcribe a normalized audio file with Whisper.cpp.",
    )
    transcribe_parser.add_argument("input", type=Path, help="Input audio path.")
    transcribe_parser.add_argument(
        "--output", type=Path, help="Transcript JSON path (defaults inside workspace)."
    )
    transcribe_parser.add_argument(
        "--language", default="auto", help="Spoken language code or auto-detect."
    )
    transcribe_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="ASR device selection (defaults to configuration).",
    )
    transcribe_parser.add_argument(
        "--threads", type=int, default=4, help="Whisper.cpp CPU thread count."
    )
    transcribe_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing transcript JSON."
    )
    transcribe_parser.add_argument(
        "--json", action="store_true", help="Emit the result paths as JSON."
    )
    srt_parser = commands.add_parser(
        "export-srt",
        help="Export a normalized transcript JSON file as SRT.",
    )
    srt_parser.add_argument("input", type=Path, help="Input transcript JSON path.")
    srt_parser.add_argument(
        "--output", type=Path, help="Output SRT path (defaults inside workspace)."
    )
    srt_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing SRT file."
    )
    srt_parser.add_argument(
        "--json", action="store_true", help="Emit the result paths as JSON."
    )
    ass_parser = commands.add_parser(
        "export-ass",
        help="Export a normalized transcript JSON file as styled ASS.",
    )
    ass_parser.add_argument("input", type=Path, help="Input transcript JSON path.")
    ass_parser.add_argument(
        "--output", type=Path, help="Output ASS path (defaults inside workspace)."
    )
    ass_parser.add_argument(
        "--style", help="Named ASS style preset (defaults to configuration)."
    )
    ass_parser.add_argument("--width", type=int, default=1920, help="ASS play width.")
    ass_parser.add_argument("--height", type=int, default=1080, help="ASS play height.")
    ass_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing ASS file."
    )
    ass_parser.add_argument(
        "--json", action="store_true", help="Emit the result paths as JSON."
    )
    preview_parser = commands.add_parser(
        "create-preview",
        help="Burn ASS subtitles into a fast, downscaled MP4 preview.",
    )
    preview_parser.add_argument("input", type=Path, help="Input video path.")
    preview_parser.add_argument("subtitles", type=Path, help="Input ASS subtitle path.")
    preview_parser.add_argument(
        "--output", type=Path, help="Output preview path (defaults inside workspace)."
    )
    preview_parser.add_argument(
        "--width", type=int, default=1280, help="Preview width; height preserves aspect ratio."
    )
    preview_parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing preview."
    )
    preview_parser.add_argument(
        "--json", action="store_true", help="Emit the result paths as JSON."
    )
    caption_parser = commands.add_parser(
        "caption",
        help="Run the complete local caption pipeline for a video.",
    )
    caption_parser.add_argument("input", type=Path, help="Input video path.")
    caption_parser.add_argument(
        "--language", default="auto", help="Spoken language code or auto-detect."
    )
    caption_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="ASR device selection (defaults to configuration).",
    )
    caption_parser.add_argument(
        "--threads", type=int, default=4, help="Whisper.cpp CPU thread count."
    )
    caption_parser.add_argument(
        "--style", help="ASS style preset (defaults to configuration)."
    )
    caption_parser.add_argument(
        "--preview-width", type=int, default=1280, help="Preview width."
    )
    caption_parser.add_argument(
        "--no-preview", action="store_true", help="Stop after transcript and subtitle exports."
    )
    caption_parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing job artifacts."
    )
    caption_parser.add_argument(
        "--json", action="store_true", help="Emit structured job results as JSON."
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

    if args.command == "transcribe":
        if config.asr.backend != "whisper_cpp":
            print(
                f"Configuration error: unsupported ASR backend '{config.asr.backend}'",
                file=sys.stderr,
            )
            return 2
        if args.threads <= 0:
            print("Configuration error: --threads must be greater than zero", file=sys.stderr)
            return 2
        output = (
            args.output
            or config.output.workspace / f"{args.input.stem}.transcript.raw.json"
        )
        if output.exists() and not args.overwrite:
            print(
                f"Output already exists: {output}; pass --overwrite to replace it",
                file=sys.stderr,
            )
            return 1
        try:
            backend = WhisperCppBackend(
                config.tools.whisper_cpp,
                config.asr.model,
            )
            transcript = backend.transcribe(
                args.input,
                TranscriptionOptions(
                    language=args.language,
                    device=args.device or config.asr.device,
                    threads=args.threads,
                ),
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(transcript.as_dict(), indent=2), encoding="utf-8"
            )
        except (VideoMcpError, ValueError, OSError) as exc:
            print(f"Transcription error: {exc}", file=sys.stderr)
            return 1
        result = {
            "input": str(args.input.resolve()),
            "transcript": str(output.resolve()),
            "language": transcript.language,
            "segments": len(transcript.segments),
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Transcript: {output.resolve()}")
            print(f"Language: {transcript.language or 'unknown'}")
            print(f"Segments: {len(transcript.segments)}")
        return 0

    if args.command == "export-srt":
        output = args.output or config.output.workspace / f"{args.input.stem}.srt"
        if output.exists() and not args.overwrite:
            print(
                f"Output already exists: {output}; pass --overwrite to replace it",
                file=sys.stderr,
            )
            return 1
        try:
            transcript = Transcript.from_dict(
                json.loads(args.input.read_text(encoding="utf-8"))
            )
            srt_path = write_srt(transcript, output, overwrite=args.overwrite)
        except (VideoMcpError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Subtitle error: {exc}", file=sys.stderr)
            return 1
        result = {
            "input": str(args.input.resolve()),
            "srt": str(srt_path),
            "cues": len(transcript.segments),
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"SRT: {srt_path}")
            print(f"Cues: {len(transcript.segments)}")
        return 0

    if args.command == "export-ass":
        output = args.output or config.output.workspace / f"{args.input.stem}.ass"
        if output.exists() and not args.overwrite:
            print(
                f"Output already exists: {output}; pass --overwrite to replace it",
                file=sys.stderr,
            )
            return 1
        try:
            if args.width <= 0 or args.height <= 0:
                raise ValueError("ASS width and height must be greater than zero")
            transcript = Transcript.from_dict(
                json.loads(args.input.read_text(encoding="utf-8"))
            )
            ass_path = write_ass(
                transcript,
                output,
                style=args.style or config.subtitles.preset,
                play_res_x=args.width,
                play_res_y=args.height,
                overwrite=args.overwrite,
            )
        except (VideoMcpError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Subtitle error: {exc}", file=sys.stderr)
            return 1
        result = {
            "input": str(args.input.resolve()),
            "ass": str(ass_path),
            "style": args.style or config.subtitles.preset,
            "cues": len(transcript.segments),
            "play_resolution": {"width": args.width, "height": args.height},
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"ASS: {ass_path}")
            print(f"Style: {result['style']}")
            print(f"Cues: {len(transcript.segments)}")
        return 0

    if args.command == "create-preview":
        output = args.output or config.output.workspace / f"{args.input.stem}.preview.mp4"
        if output.exists() and not args.overwrite:
            print(
                f"Output already exists: {output}; pass --overwrite to replace it",
                file=sys.stderr,
            )
            return 1
        try:
            preview_path = create_preview(
                args.input,
                args.subtitles,
                output,
                ffmpeg_path=config.tools.ffmpeg,
                preview_width=args.width,
                overwrite=args.overwrite,
            )
        except (VideoMcpError, ValueError, OSError) as exc:
            print(f"Render error: {exc}", file=sys.stderr)
            return 1
        result = {
            "input": str(args.input.resolve()),
            "subtitles": str(args.subtitles.resolve()),
            "preview": str(preview_path),
            "width": args.width,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Preview: {preview_path}")
            print(f"Width: {args.width}")
        return 0

    if args.command == "caption":
        if args.threads <= 0 or args.preview_width <= 0:
            print(
                "Configuration error: --threads and --preview-width must be greater than zero",
                file=sys.stderr,
            )
            return 2
        try:
            result = caption_video(
                args.input,
                config,
                CaptionOptions(
                    language=args.language,
                    device=args.device or config.asr.device,
                    threads=args.threads,
                    style=args.style or config.subtitles.preset,
                    preview_width=args.preview_width,
                    create_preview=not args.no_preview,
                    overwrite=args.overwrite,
                ),
            )
        except (VideoMcpError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Caption error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(f"Job: {result.job_dir}")
            print(f"Transcript: {result.transcript_cleaned_json}")
            print(f"SRT: {result.subtitles_srt}")
            print(f"ASS: {result.subtitles_ass}")
            if result.preview_mp4:
                print(f"Preview: {result.preview_mp4}")
            print(f"Segments: {result.segment_count}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
