"""FFmpeg subtitle rendering and preview creation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from video_mcp.errors import ExecutableNotFound, InputFileNotFound, RenderFailed

PathLike = str | Path


def render_subtitles(
    input_path: PathLike,
    subtitles_path: PathLike,
    output_path: PathLike,
    *,
    ffmpeg_path: PathLike = "ffmpeg",
    preview: bool = False,
    preview_width: int = 1280,
    overwrite: bool = False,
    timeout_seconds: float = 3600,
) -> Path:
    """Burn ASS subtitles into a new MP4 with FFmpeg.

    The subtitle path is escaped for the FFmpeg filter language separately
    from the subprocess argument list. The source video is never overwritten.
    Preview mode uses a fast H.264 encode and scales down to ``preview_width``
    while preserving aspect ratio.
    """

    source = Path(input_path).expanduser()
    subtitles = Path(subtitles_path).expanduser()
    destination = Path(output_path).expanduser()
    if not source.is_file():
        raise InputFileNotFound(f"Input video does not exist: {source}")
    if not subtitles.is_file():
        raise InputFileNotFound(f"Subtitle file does not exist: {subtitles}")
    if source.resolve() == destination.resolve():
        raise RenderFailed(
            "FFmpeg render",
            [str(ffmpeg_path), str(source), str(destination)],
            1,
            "Output video path must differ from the input video path",
        )
    if destination.exists() and not overwrite:
        raise RenderFailed(
            "FFmpeg render",
            [str(ffmpeg_path), str(source), str(destination)],
            1,
            f"Output already exists: {destination}; pass overwrite=True to replace it",
        )
    if preview and preview_width <= 0:
        raise RenderFailed(
            "FFmpeg render",
            [str(ffmpeg_path), str(source), str(destination)],
            1,
            "Preview width must be greater than zero",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"subtitles=filename='{_escape_filter_path(subtitles)}'"]
    if preview:
        filters.append(f"scale={preview_width}:-2")
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast" if preview else "medium",
        "-crf",
        "23" if preview else "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise ExecutableNotFound(
            f"FFmpeg executable was not found: {ffmpeg_path}"
        ) from exc
    except OSError as exc:
        raise ExecutableNotFound(
            f"Could not start FFmpeg '{ffmpeg_path}': {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderFailed("FFmpeg render timed out", command, -1, str(exc)) from exc

    if completed.returncode != 0:
        raise RenderFailed(
            "FFmpeg render",
            command,
            completed.returncode,
            completed.stderr.strip() or completed.stdout.strip(),
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RenderFailed(
            "FFmpeg render",
            command,
            0,
            f"FFmpeg completed without creating a usable video: {destination}",
        )
    return destination.resolve()


def create_preview(
    input_path: PathLike,
    subtitles_path: PathLike,
    output_path: PathLike,
    *,
    ffmpeg_path: PathLike = "ffmpeg",
    preview_width: int = 1280,
    overwrite: bool = False,
    timeout_seconds: float = 3600,
) -> Path:
    """Create a fast, downscaled burned-in subtitle preview."""

    return render_subtitles(
        input_path,
        subtitles_path,
        output_path,
        ffmpeg_path=ffmpeg_path,
        preview=True,
        preview_width=preview_width,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )


def _escape_filter_path(path: Path) -> str:
    """Escape a Windows path for FFmpeg's ``subtitles`` filter argument."""

    value = path.resolve().as_posix()
    value = value.replace("\\", "\\\\")
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    return value
