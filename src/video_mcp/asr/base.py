"""ASR backend interfaces and shared transcription options."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from video_mcp.models import Transcript


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    """Options shared by local speech-recognition backends."""

    language: str = "auto"
    device: str = "auto"
    threads: int = 4
    translate: bool = False
    timeout_seconds: float = 3600
    process_started: Callable[[int], None] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.threads <= 0:
            raise ValueError("threads must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")


class ASRBackend(ABC):
    """Interface implemented by each local transcription engine."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        options: TranscriptionOptions | None = None,
    ) -> Transcript:
        """Transcribe an audio file into normalized transcript data."""
