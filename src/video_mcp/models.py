"""Normalized application data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    """The video properties needed by the captioning pipeline."""

    codec: str | None
    width: int | None
    height: int | None
    frame_rate: float | None
    rotation: int | None


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """The audio properties needed to select and extract speech audio."""

    codec: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """A stable, application-owned summary of an input media file."""

    path: Path
    duration_ms: int | None
    format_name: str | None
    size_bytes: int | None
    bit_rate: int | None
    video: VideoStreamInfo | None
    audio: AudioStreamInfo | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the media summary."""

        value = asdict(self)
        value["path"] = str(self.path)
        return value


@dataclass(frozen=True, slots=True)
class Word:
    """A timestamped piece of recognized speech."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SubtitleSegment:
    """A normalized caption-sized unit from an ASR backend."""

    id: str
    start_ms: int
    end_ms: int
    text: str
    words: list[Word]
    speaker: str | None = None


@dataclass(frozen=True, slots=True)
class Transcript:
    """Versioned, JSON-serializable transcript data."""

    language: str | None
    duration_ms: int
    segments: list[SubtitleSegment]
    schema_version: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by later pipeline steps."""

        return asdict(self)
