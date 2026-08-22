"""FFprobe integration and normalized media metadata parsing."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from video_mcp.errors import (
    ExecutableNotFound,
    InputFileNotFound,
    InvalidMediaProbeOutput,
    MediaProbeFailed,
    UnsupportedMedia,
)
from video_mcp.models import AudioStreamInfo, MediaInfo, VideoStreamInfo

PathLike = str | Path


def probe_video(
    input_path: PathLike,
    *,
    ffprobe_path: PathLike = "ffprobe",
    timeout_seconds: float = 60,
) -> MediaInfo:
    """Inspect a video with FFprobe and return normalized metadata.

    Arguments are passed directly to ``subprocess.run`` so Windows paths with
    spaces remain a single argument and no shell parsing is involved.
    """

    source = Path(input_path).expanduser()
    if not source.is_file():
        raise InputFileNotFound(f"Input media file does not exist: {source}")

    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
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
        raise ExecutableNotFound(f"FFprobe executable was not found: {ffprobe_path}") from exc
    except OSError as exc:
        raise ExecutableNotFound(f"Could not start FFprobe '{ffprobe_path}': {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeFailed(
            "FFprobe inspection timed out",
            command,
            -1,
            str(exc),
        ) from exc

    if completed.returncode != 0:
        raise MediaProbeFailed(
            "FFprobe inspection",
            command,
            completed.returncode,
            completed.stderr,
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidMediaProbeOutput(
            f"FFprobe returned invalid JSON for {source}: {exc.msg}"
        ) from exc

    try:
        return parse_ffprobe_output(payload, source)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidMediaProbeOutput(
            f"FFprobe returned an unexpected media structure for {source}: {exc}"
        ) from exc


def parse_ffprobe_output(payload: str | Mapping[str, Any], input_path: PathLike) -> MediaInfo:
    """Convert FFprobe JSON into the application's stable media model."""

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InvalidMediaProbeOutput(f"Invalid FFprobe JSON: {exc.msg}") from exc
    else:
        data = payload

    if not isinstance(data, Mapping):
        raise InvalidMediaProbeOutput("FFprobe JSON root must be an object")

    streams = data.get("streams", [])
    if not isinstance(streams, list):
        raise InvalidMediaProbeOutput("FFprobe streams must be an array")

    video_stream = _first_stream(streams, "video")
    if video_stream is None:
        raise UnsupportedMedia(f"No video stream found in {input_path}")
    audio_stream = _first_stream(streams, "audio")

    format_data = data.get("format", {})
    if not isinstance(format_data, Mapping):
        raise InvalidMediaProbeOutput("FFprobe format must be an object")

    duration_ms = _duration_ms(format_data.get("duration"))
    if duration_ms is None:
        stream_durations = [
            parsed_duration
            for stream in streams
            if isinstance(stream, Mapping)
            for parsed_duration in [_duration_ms(stream.get("duration"))]
            if parsed_duration is not None
        ]
        duration_ms = max(stream_durations, default=None)

    return MediaInfo(
        path=Path(input_path).expanduser().resolve(),
        duration_ms=duration_ms,
        format_name=_optional_string(format_data.get("format_name")),
        size_bytes=_optional_int(format_data.get("size")),
        bit_rate=_optional_int(format_data.get("bit_rate")),
        video=VideoStreamInfo(
            codec=_optional_string(video_stream.get("codec_name")),
            width=_optional_int(video_stream.get("width")),
            height=_optional_int(video_stream.get("height")),
            frame_rate=_frame_rate(video_stream),
            rotation=_rotation(video_stream),
        ),
        audio=(
            AudioStreamInfo(
                codec=_optional_string(audio_stream.get("codec_name")),
                sample_rate=_optional_int(audio_stream.get("sample_rate")),
                channels=_optional_int(audio_stream.get("channels")),
                channel_layout=_optional_string(audio_stream.get("channel_layout")),
            )
            if audio_stream is not None
            else None
        ),
    )


def format_media_info(info: MediaInfo) -> str:
    """Format a compact human-readable media inspection report."""

    lines = [f"Input: {info.path}"]
    lines.append(f"Duration: {_format_duration(info.duration_ms)}")
    lines.append(f"Format: {info.format_name or 'unknown'}")
    if info.video is not None:
        video = info.video
        dimensions = (
            f"{video.width}x{video.height}"
            if video.width is not None and video.height is not None
            else "unknown size"
        )
        rate = f"{video.frame_rate:.3f} fps" if video.frame_rate is not None else "unknown fps"
        rotation = f", rotation {video.rotation}°" if video.rotation is not None else ""
        lines.append(f"Video: {video.codec or 'unknown'}, {dimensions}, {rate}{rotation}")
    if info.audio is not None:
        audio = info.audio
        sample_rate = (
            f"{audio.sample_rate} Hz" if audio.sample_rate is not None else "unknown sample rate"
        )
        channels = f", {audio.channels} channel(s)" if audio.channels is not None else ""
        lines.append(f"Audio: {audio.codec or 'unknown'}, {sample_rate}{channels}")
    else:
        lines.append("Audio: none")
    return "\n".join(lines)


def _first_stream(streams: list[Any], codec_type: str) -> Mapping[str, Any] | None:
    for stream in streams:
        if isinstance(stream, Mapping) and stream.get("codec_type") == codec_type:
            return stream
    return None


def _optional_string(value: Any) -> str | None:
    if value is None or value == "N/A":
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "N/A" or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _duration_ms(value: Any) -> int | None:
    if value is None or value == "N/A" or value == "":
        return None
    try:
        return round(float(value) * 1000)
    except (TypeError, ValueError):
        return None


def _frame_rate(stream: Mapping[str, Any]) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if value in (None, "", "N/A", "0/0"):
            continue
        try:
            if isinstance(value, str) and "/" in value:
                numerator, denominator = value.split("/", 1)
                denominator_value = float(denominator)
                if denominator_value == 0:
                    continue
                return float(numerator) / denominator_value
            return float(value)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return None


def _rotation(stream: Mapping[str, Any]) -> int | None:
    side_data = stream.get("side_data_list", [])
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, Mapping) and entry.get("rotation") is not None:
                return _optional_int(entry.get("rotation"))

    tags = stream.get("tags", {})
    if isinstance(tags, Mapping):
        return _optional_int(tags.get("rotate"))
    return None


def _format_duration(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "unknown"
    total_seconds, milliseconds = divmod(duration_ms, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
