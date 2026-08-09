"""Application service for local ASR transcription."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr.factory import create_asr_backend
from video_mcp.config import AppConfig

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Path and summary returned by a transcription job."""

    input_path: Path
    transcript_path: Path
    language: str | None
    segment_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable transcription result."""

        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, Path):
                value[key] = str(item)
        return value


def transcribe_audio(
    input_path: PathLike,
    config: AppConfig,
    *,
    output_path: PathLike | None = None,
    language: str = "auto",
    device: str | None = None,
    threads: int = 4,
    overwrite: bool = False,
) -> TranscriptionResult:
    """Transcribe normalized audio and persist the versioned transcript JSON."""

    audio = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser() if output_path else (
        config.output.workspace / f"{audio.stem}.transcript.raw.json"
    )
    destination = destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}; pass overwrite=True to replace it"
        )

    backend = create_asr_backend(config)
    transcript = backend.transcribe(
        audio,
        TranscriptionOptions(
            language=language,
            device=device or config.asr.device,
            threads=threads,
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(transcript.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return TranscriptionResult(
        input_path=audio,
        transcript_path=destination,
        language=transcript.language,
        segment_count=len(transcript.segments),
    )
