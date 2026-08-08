"""FFmpeg integrations used by the local captioning pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

from video_mcp.errors import (
    AudioExtractionFailed,
    ExecutableNotFound,
    InputFileNotFound,
)

PathLike = str | Path


def extract_audio(
    input_path: PathLike,
    output_path: PathLike,
    *,
    ffmpeg_path: PathLike = "ffmpeg",
    overwrite: bool = False,
    timeout_seconds: float = 600,
) -> Path:
    """Extract the first audio stream as mono 16 kHz PCM WAV.

    Existing output is preserved unless ``overwrite=True`` is explicitly
    requested. The source file is never modified.
    """

    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    if not source.is_file():
        raise InputFileNotFound(f"Input media file does not exist: {source}")
    if source.resolve() == destination.resolve():
        raise AudioExtractionFailed(
            "Audio extraction",
            [str(ffmpeg_path), str(source), str(destination)],
            1,
            "Output audio path must differ from the input media path",
        )
    if destination.exists() and not overwrite:
        raise AudioExtractionFailed(
            "Audio extraction",
            [str(ffmpeg_path), str(source), str(destination)],
            1,
            f"Output already exists: {destination}; pass overwrite=True to replace it",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
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
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
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
        raise AudioExtractionFailed(
            "Audio extraction timed out",
            command,
            -1,
            str(exc),
        ) from exc

    if completed.returncode != 0:
        raise AudioExtractionFailed(
            "Audio extraction",
            command,
            completed.returncode,
            completed.stderr,
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise AudioExtractionFailed(
            "Audio extraction",
            command,
            0,
            f"FFmpeg completed without creating a usable WAV file: {destination}",
        )
    return destination.resolve()
