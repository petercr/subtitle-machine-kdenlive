"""Run a small local-tool smoke test without downloading ML models."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from video_mcp.config import load_config
from video_mcp.errors import VideoMcpError
from video_mcp.media.ffmpeg import extract_audio
from video_mcp.media.probe import probe_video
from video_mcp.media.render import create_preview
from video_mcp.models import SubtitleSegment, Transcript
from video_mcp.subtitles.ass import write_ass
from video_mcp.subtitles.srt import write_srt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise FFprobe, FFmpeg, subtitle exports, and preview rendering."
    )
    parser.add_argument(
        "--config", type=Path, help="Path to video-mcp.yaml or the example config."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work") / "smoke-test",
        help="Directory for generated smoke-test artifacts.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Use an existing video instead of generating a synthetic clip.",
    )
    return parser


def _generate_video(path: Path, ffmpeg_path: Path) -> None:
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x180:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:sample_rate=44100",
        "-t",
        "2",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Could not generate smoke-test video: {diagnostic}")


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_ms=2000,
        segments=[
            SubtitleSegment("smoke-1", 0, 1000, "Hello smoke test.", []),
            SubtitleSegment("smoke-2", 1000, 2000, "The local tools work.", []),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.input.expanduser().resolve() if args.input else output / "source.mp4"
    if args.input is None:
        _generate_video(source, config.tools.ffmpeg)
    elif not source.is_file():
        raise FileNotFoundError(f"Smoke-test input does not exist: {source}")

    media = probe_video(source, ffprobe_path=config.tools.ffprobe)
    audio = extract_audio(
        source,
        output / "audio.wav",
        ffmpeg_path=config.tools.ffmpeg,
        overwrite=True,
    )
    transcript = _transcript()
    transcript_path = output / "transcript.cleaned.json"
    transcript_path.write_text(
        json.dumps(transcript.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    srt = write_srt(transcript, output / "subtitles.srt", overwrite=True)
    ass = write_ass(
        transcript,
        output / "subtitles.ass",
        style=config.subtitles.preset,
        overwrite=True,
    )
    preview = create_preview(
        source,
        ass,
        output / "preview.mp4",
        ffmpeg_path=config.tools.ffmpeg,
        preview_width=320,
        overwrite=True,
    )

    print(
        json.dumps(
            {
                "source": str(source),
                "audio": str(audio),
                "transcript": str(transcript_path),
                "srt": str(srt),
                "ass": str(ass),
                "preview": str(preview),
                "duration_ms": media.duration_ms,
                "video_width": media.video.width if media.video else None,
                "video_height": media.video.height if media.video else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VideoMcpError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Smoke test failed: {exc}") from exc
