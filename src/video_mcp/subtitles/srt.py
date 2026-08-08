"""Deterministic SubRip (SRT) subtitle generation."""

from __future__ import annotations

from pathlib import Path

from video_mcp.errors import SubtitleGenerationFailed
from video_mcp.models import SubtitleSegment, Transcript

PathLike = str | Path


def format_srt_timestamp(milliseconds: int) -> str:
    """Format milliseconds as the SRT ``HH:MM:SS,mmm`` timestamp format."""

    if milliseconds < 0:
        raise SubtitleGenerationFailed("SRT timestamps cannot be negative")
    total_seconds, millis = divmod(milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def generate_srt(transcript: Transcript) -> str:
    """Render a transcript as stable UTF-8-ready SRT text.

    Cue numbers are generated from the ordered transcript list. Input segment
    IDs remain in the JSON source of truth and are not emitted into SRT.
    """

    cues: list[str] = []
    previous_end = 0
    for cue_number, segment in enumerate(transcript.segments, start=1):
        _validate_segment(segment, cue_number, previous_end)
        text = segment.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            raise SubtitleGenerationFailed(f"SRT cue {cue_number} has no text")
        cues.append(
            "\n".join(
                (
                    str(cue_number),
                    f"{format_srt_timestamp(segment.start_ms)} --> "
                    f"{format_srt_timestamp(segment.end_ms)}",
                    text,
                )
            )
        )
        previous_end = segment.end_ms
    return "\n\n".join(cues) + ("\n" if cues else "")


def write_srt(
    transcript: Transcript,
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Write SRT text without replacing an existing file by default."""

    destination = Path(output_path).expanduser()
    if destination.exists() and not overwrite:
        raise SubtitleGenerationFailed(
            f"Output already exists: {destination}; pass overwrite=True to replace it"
        )
    content = generate_srt(transcript)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return destination.resolve()


def _validate_segment(
    segment: SubtitleSegment, cue_number: int, previous_end: int
) -> None:
    if segment.start_ms < 0 or segment.end_ms < 0:
        raise SubtitleGenerationFailed(f"SRT cue {cue_number} has a negative timestamp")
    if segment.end_ms <= segment.start_ms:
        raise SubtitleGenerationFailed(
            f"SRT cue {cue_number} must end after it starts"
        )
    if segment.start_ms < previous_end:
        raise SubtitleGenerationFailed(f"SRT cue {cue_number} overlaps the previous cue")
