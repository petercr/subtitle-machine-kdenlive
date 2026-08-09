"""Thin stdio MCP interface over the application services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from video_mcp import __version__
from video_mcp.config import AppConfig, load_config
from video_mcp.media.probe import probe_video
from video_mcp.media.render import create_preview
from video_mcp.models import Transcript
from video_mcp.services.captioning import CaptionOptions, caption_video
from video_mcp.services.kdenlive import create_kdenlive_project
from video_mcp.services.transcription import transcribe_audio
from video_mcp.subtitles.ass import write_ass
from video_mcp.subtitles.srt import write_srt

PathLike = str | Path

mcp = MCPServer(
    name="video-mcp",
    title="Video Subtitle MCP",
    description="Local-first video captioning, subtitle, preview, and Kdenlive tools.",
    instructions=(
        "Use these tools for deterministic local media processing. "
        "No cloud APIs or Kdenlive GUI automation are required."
    ),
    version=__version__,
)


def _config(config_path: str | None) -> AppConfig:
    return load_config(config_path)


def _read_transcript(path: PathLike) -> Transcript:
    transcript_path = Path(path).expanduser().resolve()
    return Transcript.from_dict(
        json.loads(transcript_path.read_text(encoding="utf-8"))
    )


@mcp.tool(
    name="video.inspect",
    description="Inspect a video with FFprobe and return normalized media metadata.",
    structured_output=True,
)
def video_inspect(
    input_path: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    config = _config(config_path)
    return probe_video(input_path, ffprobe_path=config.tools.ffprobe).as_dict()


@mcp.tool(
    name="video.transcribe",
    description="Transcribe normalized audio with the configured local ASR backend.",
    structured_output=True,
)
def video_transcribe(
    audio_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    language: str = "auto",
    device: str | None = None,
    threads: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    result = transcribe_audio(
        audio_path,
        _config(config_path),
        output_path=output_path,
        language=language,
        device=device,
        threads=threads,
        overwrite=overwrite,
    )
    return result.as_dict()


@mcp.tool(
    name="video.caption",
    description="Run the complete local caption pipeline for a video.",
    structured_output=True,
)
def video_caption(
    input_path: str,
    config_path: str | None = None,
    language: str = "auto",
    device: str | None = None,
    threads: int = 4,
    style: str | None = None,
    preview_width: int = 1280,
    create_preview_output: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = _config(config_path)
    result = caption_video(
        input_path,
        config,
        CaptionOptions(
            language=language,
            device=device or config.asr.device,
            threads=threads,
            style=style or config.subtitles.preset,
            preview_width=preview_width,
            create_preview=create_preview_output,
            overwrite=overwrite,
        ),
    )
    return result.as_dict()


def _create_preview(
    input_path: str,
    subtitles_path: str,
    config_path: str | None,
    output_path: str | None,
    preview_width: int,
    overwrite: bool,
) -> dict[str, Any]:
    config = _config(config_path)
    source = Path(input_path).expanduser().resolve()
    subtitles = Path(subtitles_path).expanduser().resolve()
    destination = Path(output_path).expanduser() if output_path else (
        config.output.workspace / f"{source.stem}.preview.mp4"
    )
    preview = create_preview(
        source,
        subtitles,
        destination,
        ffmpeg_path=config.tools.ffmpeg,
        preview_width=preview_width,
        overwrite=overwrite,
    )
    return {
        "input_path": str(source),
        "subtitles_path": str(subtitles),
        "preview_path": str(preview),
        "width": preview_width,
    }


@mcp.tool(
    name="video.create_preview",
    description="Burn ASS subtitles into a fast, downscaled MP4 preview.",
    structured_output=True,
)
def video_create_preview(
    input_path: str,
    subtitles_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    preview_width: int = 1280,
    overwrite: bool = False,
) -> dict[str, Any]:
    return _create_preview(
        input_path,
        subtitles_path,
        config_path,
        output_path,
        preview_width,
        overwrite,
    )


@mcp.tool(
    name="video.render",
    description="Render a burned-in ASS subtitle preview; alias for video.create_preview.",
    structured_output=True,
)
def video_render(
    input_path: str,
    subtitles_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    preview_width: int = 1280,
    overwrite: bool = False,
) -> dict[str, Any]:
    return _create_preview(
        input_path,
        subtitles_path,
        config_path,
        output_path,
        preview_width,
        overwrite,
    )


@mcp.tool(
    name="subtitle.export_srt",
    description="Export a normalized transcript JSON file as SRT.",
    structured_output=True,
)
def subtitle_export_srt(
    transcript_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = _config(config_path)
    source = Path(transcript_path).expanduser().resolve()
    destination = Path(output_path).expanduser() if output_path else (
        config.output.workspace / f"{source.stem}.srt"
    )
    transcript = _read_transcript(source)
    result = write_srt(transcript, destination, overwrite=overwrite)
    return {"input_path": str(source), "srt_path": str(result), "cues": len(transcript.segments)}


@mcp.tool(
    name="subtitle.export_ass",
    description="Export a normalized transcript JSON file as styled ASS.",
    structured_output=True,
)
def subtitle_export_ass(
    transcript_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    style: str | None = None,
    width: int = 1920,
    height: int = 1080,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = _config(config_path)
    source = Path(transcript_path).expanduser().resolve()
    destination = Path(output_path).expanduser() if output_path else (
        config.output.workspace / f"{source.stem}.ass"
    )
    transcript = _read_transcript(source)
    result = write_ass(
        transcript,
        destination,
        style=style or config.subtitles.preset,
        play_res_x=width,
        play_res_y=height,
        overwrite=overwrite,
    )
    return {
        "input_path": str(source),
        "ass_path": str(result),
        "style": style or config.subtitles.preset,
        "cues": len(transcript.segments),
        "play_resolution": {"width": width, "height": height},
    }


@mcp.tool(
    name="project.create_kdenlive",
    description="Create an editable Kdenlive project from a video and SRT file.",
    structured_output=True,
)
def project_create_kdenlive(
    input_path: str,
    subtitles_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    result = create_kdenlive_project(
        input_path,
        subtitles_path,
        _config(config_path),
        output_path=output_path,
        overwrite=overwrite,
    )
    return result.as_dict()


def main() -> None:
    """Run the MCP server over its default stdio transport."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
