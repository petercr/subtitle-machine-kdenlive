"""Parakeet ASR backend through the optional whisper.cpp CLI."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from video_mcp.asr.base import ASRBackend, TranscriptionOptions
from video_mcp.errors import (
    ExecutableNotFound,
    InputFileNotFound,
    InvalidTranscriptOutput,
    ModelNotFound,
    TranscriptionFailed,
)
from video_mcp.models import SubtitleSegment, Transcript, Word

logger = logging.getLogger(__name__)
PathLike = str | Path

_SEGMENT_RE = re.compile(
    r'^\s*Segment\s+\d+:\s+\[(?P<start>-?\d+)\s+->\s+(?P<end>-?\d+)\]\s+"(?P<text>.*)"\s*$'
)
_TOKEN_RE = re.compile(
    r'^\s*\[\s*\d+\]\s+.*?\bp=(?P<prob>-?[0-9.]+)\s+'
    r'.*?\bt0=\s*(?P<start>-?\d+)\s+t1=\s*(?P<end>-?\d+)\s+'
    r'word_start=(?P<word_start>true|false)\s+"(?P<text>.*)"\s*$'
)


@dataclass
class _ParsedSegment:
    """Mutable parser state for one CLI segment."""

    start_frame: int
    end_frame: int
    text: str
    words: list[Word] = field(default_factory=list)
    pending_word: str = ""
    pending_start_ms: int | None = None
    pending_end_ms: int | None = None
    pending_confidence: float | None = None

    def add_token(
        self,
        text: str,
        start_ms: int,
        end_ms: int,
        confidence: float,
        is_word_start: bool,
    ) -> None:
        token_text = text.replace("▁", "")
        if is_word_start and self.pending_word:
            self._flush_word()
        if is_word_start:
            self.pending_word = token_text
            self.pending_start_ms = start_ms
            self.pending_end_ms = end_ms
            self.pending_confidence = confidence
        elif self.pending_word:
            self.pending_word += token_text
            self.pending_end_ms = max(self.pending_end_ms or end_ms, end_ms)
            self.pending_confidence = min(
                self.pending_confidence
                if self.pending_confidence is not None
                else confidence,
                confidence,
            )

    def finish(self) -> None:
        self._flush_word()

    def _flush_word(self) -> None:
        if not self.pending_word or self.pending_start_ms is None or self.pending_end_ms is None:
            self.pending_word = ""
            return
        self.words.append(
            Word(
                start_ms=self.pending_start_ms,
                end_ms=max(self.pending_start_ms, self.pending_end_ms),
                text=self.pending_word,
                confidence=self.pending_confidence,
            )
        )
        self.pending_word = ""
        self.pending_start_ms = None
        self.pending_end_ms = None
        self.pending_confidence = None


class ParakeetBackend(ASRBackend):
    """Run the optional ``parakeet-cli`` executable and normalize its output.

    The current whisper.cpp Parakeet CLI writes segment and token diagnostics
    rather than JSON. The adapter requests that stable diagnostic format and
    converts its 10 ms audio-frame timestamps into the application model.
    """

    def __init__(self, executable: PathLike, model: PathLike) -> None:
        self.executable = Path(executable).expanduser()
        self.model = Path(model).expanduser()

    def transcribe(
        self,
        audio_path: PathLike,
        options: TranscriptionOptions | None = None,
    ) -> Transcript:
        options = options or TranscriptionOptions()
        audio = Path(audio_path).expanduser()
        if not audio.is_file():
            raise InputFileNotFound(f"Audio file does not exist: {audio}")
        if not self.model.is_file():
            raise ModelNotFound(f"Parakeet model does not exist: {self.model}")

        command = self._command(audio, options)
        completed = self._run(command, options)
        if completed.returncode != 0 and options.device == "auto":
            logger.warning("Parakeet auto device invocation failed; retrying on CPU")
            command = self._command(audio, options, force_cpu=True)
            completed = self._run(command, options)
        if completed.returncode != 0:
            raise TranscriptionFailed(
                "Parakeet transcription",
                command,
                completed.returncode,
                _diagnostic_output(completed),
            )

        return parse_parakeet_output(
            f"{completed.stdout}\n{completed.stderr}", audio
        )

    def _command(
        self,
        audio: Path,
        options: TranscriptionOptions,
        *,
        force_cpu: bool = False,
    ) -> list[str]:
        command = [
            str(self.executable),
            "-m",
            str(self.model),
            "-f",
            str(audio),
            "-ps",
            "-np",
            "-t",
            str(options.threads),
        ]
        if options.device == "cpu" or force_cpu:
            command.append("-ng")
        return command

    def _run(
        self, command: list[str], options: TranscriptionOptions
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=options.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ExecutableNotFound(
                f"Parakeet executable was not found: {self.executable}"
            ) from exc
        except OSError as exc:
            raise ExecutableNotFound(
                f"Could not start Parakeet '{self.executable}': {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionFailed(
                "Parakeet transcription timed out",
                command,
                -1,
                str(exc),
            ) from exc


def parse_parakeet_output(payload: str, audio_path: PathLike) -> Transcript:
    """Normalize ``parakeet-cli --print-segments`` output."""

    if not isinstance(payload, str):
        raise InvalidTranscriptOutput("Parakeet output must be text")

    parsed: list[_ParsedSegment] = []
    current: _ParsedSegment | None = None
    for line in payload.splitlines():
        segment_match = _SEGMENT_RE.match(line)
        if segment_match:
            if current is not None:
                current.finish()
                parsed.append(current)
            start_frame = int(segment_match.group("start"))
            end_frame = int(segment_match.group("end"))
            if start_frame < 0 or end_frame < start_frame:
                raise InvalidTranscriptOutput(
                    f"Parakeet segment has invalid timestamps for {audio_path}"
                )
            current = _ParsedSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                text=segment_match.group("text").strip(),
            )
            continue

        token_match = _TOKEN_RE.match(line)
        if token_match and current is not None:
            start_frame = int(token_match.group("start"))
            end_frame = int(token_match.group("end"))
            if start_frame < 0 or end_frame < start_frame:
                raise InvalidTranscriptOutput(
                    f"Parakeet token has invalid timestamps for {audio_path}"
                )
            current.add_token(
                token_match.group("text"),
                start_frame * 10,
                end_frame * 10,
                float(token_match.group("prob")),
                token_match.group("word_start") == "true",
            )

    if current is not None:
        current.finish()
        parsed.append(current)
    if not parsed:
        raise InvalidTranscriptOutput(
            f"Parakeet produced no segment output for {audio_path}"
        )

    segments = [
        SubtitleSegment(
            id=f"seg-{index:04d}",
            start_ms=segment.start_frame * 10,
            end_ms=segment.end_frame * 10,
            text=segment.text,
            words=segment.words,
        )
        for index, segment in enumerate(parsed, start=1)
        if segment.text
    ]
    if not segments:
        raise InvalidTranscriptOutput(
            f"Parakeet produced no non-empty segments for {audio_path}"
        )
    return Transcript(
        language=None,
        duration_ms=max(segment.end_ms for segment in segments),
        segments=segments,
    )


def _diagnostic_output(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    return stderr or stdout or "no diagnostic output"
