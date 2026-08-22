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

    @classmethod
    def from_dict(cls, value: Any) -> Transcript:
        """Load the stable JSON representation written by the ASR pipeline."""

        if not isinstance(value, dict):
            raise ValueError("Transcript JSON root must be an object")
        schema_version = int(value.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(f"Unsupported transcript schema version: {schema_version}")
        raw_segments = value.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("Transcript segments must be an array")

        segments: list[SubtitleSegment] = []
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                raise ValueError("Transcript segment must be an object")
            raw_words = raw_segment.get("words", [])
            if not isinstance(raw_words, list):
                raise ValueError("Transcript words must be an array")
            words = [
                Word(
                    start_ms=int(raw_word["start_ms"]),
                    end_ms=int(raw_word["end_ms"]),
                    text=str(raw_word["text"]),
                    confidence=(
                        float(raw_word["confidence"])
                        if raw_word.get("confidence") is not None
                        else None
                    ),
                )
                for raw_word in raw_words
                if isinstance(raw_word, dict)
            ]
            if len(words) != len(raw_words):
                raise ValueError("Transcript word must be an object")
            segments.append(
                SubtitleSegment(
                    id=str(raw_segment["id"]),
                    start_ms=int(raw_segment["start_ms"]),
                    end_ms=int(raw_segment["end_ms"]),
                    text=str(raw_segment["text"]),
                    words=words,
                    speaker=(
                        str(raw_segment["speaker"])
                        if raw_segment.get("speaker") is not None
                        else None
                    ),
                )
            )
        return cls(
            language=(str(value["language"]) if value.get("language") is not None else None),
            duration_ms=int(value.get("duration_ms", 0)),
            segments=segments,
            schema_version=schema_version,
        )
