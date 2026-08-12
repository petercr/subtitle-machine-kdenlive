"""Transcript cleanup orchestration with a deterministic fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from video_mcp.config import AppConfig
from video_mcp.errors import VideoMcpError
from video_mcp.models import Transcript
from video_mcp.subtitles.cleaner import DeterministicCleaner, LocalLLMCleaner
from video_mcp.subtitles.formatter import format_transcript

logger = logging.getLogger(__name__)
PathLike = str | Path


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Result of applying one cleanup policy to a transcript."""

    transcript: Transcript
    used_llm: bool
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class CleanupFileResult:
    """Paths and summary returned by transcript-file cleanup."""

    input_path: Path
    output_path: Path
    segment_count: int
    used_llm: bool
    warnings: list[str]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["input_path"] = str(self.input_path)
        value["output_path"] = str(self.output_path)
        return value


def clean_transcript(transcript: Transcript, config: AppConfig) -> CleanupResult:
    """Clean a transcript, falling back safely when the optional LLM fails."""

    deterministic = DeterministicCleaner()
    if not config.llm.enabled:
        cleaned = deterministic.clean(transcript)
        return CleanupResult(
            transcript=_format(cleaned, config),
            used_llm=False,
            warnings=["Local LLM cleanup is disabled; deterministic cleanup was used."],
        )

    cleaner = LocalLLMCleaner(
        config.tools.llama_cpp,
        config.llm.model,
        max_segments_per_chunk=config.llm.max_segments_per_chunk,
        max_chars_per_chunk=config.llm.max_chars_per_chunk,
        max_tokens=config.llm.max_tokens,
    )
    try:
        cleaned = cleaner.clean(transcript)
        return CleanupResult(
            transcript=_format(cleaned, config),
            used_llm=True,
            warnings=[],
        )
    except (VideoMcpError, ValueError) as exc:
        logger.warning("Local LLM cleanup failed; using deterministic fallback: %s", exc)
        return CleanupResult(
            transcript=_format(deterministic.clean(transcript), config),
            used_llm=False,
            warnings=[f"Local LLM cleanup failed; deterministic fallback was used: {exc}"],
        )


def _format(transcript: Transcript, config: AppConfig) -> Transcript:
    return format_transcript(
        transcript,
        max_chars=config.subtitles.max_chars_per_line * config.subtitles.max_lines,
        max_chars_per_line=config.subtitles.max_chars_per_line,
    )


def clean_transcript_file(
    input_path: PathLike,
    config: AppConfig,
    *,
    output_path: PathLike | None = None,
    overwrite: bool = False,
) -> CleanupFileResult:
    """Read, clean, and write a normalized transcript JSON file."""

    source = Path(input_path).expanduser().resolve()
    destination = (
        Path(output_path).expanduser()
        if output_path
        else config.output.workspace / f"{source.stem}.cleaned.json"
    ).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}; pass overwrite=True to replace it"
        )
    transcript = Transcript.from_dict(json.loads(source.read_text(encoding="utf-8")))
    result = clean_transcript(transcript, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.transcript.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return CleanupFileResult(
        input_path=source,
        output_path=destination,
        segment_count=len(result.transcript.segments),
        used_llm=result.used_llm,
        warnings=result.warnings,
    )
