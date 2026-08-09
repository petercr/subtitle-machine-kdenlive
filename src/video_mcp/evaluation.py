"""Small, dependency-free ASR comparison helpers."""

from __future__ import annotations

import re
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr.factory import create_asr_backend
from video_mcp.config import AppConfig
from video_mcp.models import Transcript

PathLike = str | Path
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ASRBenchmarkResult:
    """Measured output from one ASR backend run."""

    backend: str
    input_path: Path
    audio_duration_seconds: float | None
    elapsed_seconds: float
    real_time_factor: float | None
    segment_count: int | None
    word_count: int | None
    language: str | None
    word_error_rate: float | None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable benchmark record."""

        value = asdict(self)
        value["input_path"] = str(self.input_path)
        return value


def benchmark_asr(
    input_path: PathLike,
    config: AppConfig,
    *,
    reference_text: str | None = None,
    language: str = "auto",
    device: str | None = None,
    threads: int = 4,
) -> ASRBenchmarkResult:
    """Run one configured backend and measure wall-clock performance."""

    audio = Path(input_path).expanduser().resolve()
    duration = audio_duration_seconds(audio)
    backend_name = config.asr.backend
    started = time.perf_counter()
    try:
        transcript = create_asr_backend(config).transcribe(
            audio,
            TranscriptionOptions(
                language=language,
                device=device or config.asr.device,
                threads=threads,
            ),
        )
    except Exception as exc:  # benchmark records failures for comparison
        elapsed = time.perf_counter() - started
        return ASRBenchmarkResult(
            backend=backend_name,
            input_path=audio,
            audio_duration_seconds=duration,
            elapsed_seconds=elapsed,
            real_time_factor=_real_time_factor(elapsed, duration),
            segment_count=None,
            word_count=None,
            language=None,
            word_error_rate=None,
            error=str(exc),
        )

    elapsed = time.perf_counter() - started
    hypothesis = transcript_text(transcript)
    return ASRBenchmarkResult(
        backend=backend_name,
        input_path=audio,
        audio_duration_seconds=duration,
        elapsed_seconds=elapsed,
        real_time_factor=_real_time_factor(elapsed, duration),
        segment_count=len(transcript.segments),
        word_count=len(normalize_words(hypothesis)),
        language=transcript.language,
        word_error_rate=(
            word_error_rate(reference_text, hypothesis)
            if reference_text is not None
            else None
        ),
    )


def audio_duration_seconds(path: PathLike) -> float | None:
    """Read duration for a normalized PCM WAV without external dependencies."""

    try:
        with wave.open(str(Path(path)), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                return None
            return audio.getnframes() / frame_rate
    except (OSError, wave.Error):
        return None


def normalize_words(text: str) -> list[str]:
    """Normalize text for a case- and punctuation-insensitive WER comparison."""

    return [word.lower() for word in _WORD_RE.findall(text)]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate word error rate using Levenshtein edit distance."""

    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous = list(range(len(hypothesis_words) + 1))
    for reference_word in reference_words:
        current = [previous[0] + 1]
        for index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous[index - 1] + (reference_word != hypothesis_word)
            insertion = current[index - 1] + 1
            deletion = previous[index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference_words)


def _real_time_factor(elapsed: float, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    return elapsed / duration


def transcript_text(transcript: Transcript) -> str:
    """Return the plain text used for benchmark comparisons."""

    return " ".join(segment.text for segment in transcript.segments)
