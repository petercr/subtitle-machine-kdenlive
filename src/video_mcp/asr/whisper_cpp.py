"""Whisper.cpp ASR backend."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


class WhisperCppBackend(ASRBackend):
    """Run the installed whisper-cli executable and parse its JSON output."""

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
            raise ModelNotFound(f"Whisper model does not exist: {self.model}")

        with tempfile.TemporaryDirectory(prefix="video-mcp-whisper-") as temp_dir:
            output_base = Path(temp_dir) / "transcript"
            command = self._command(audio, output_base, options)
            completed = self._run(command, options)
            if completed.returncode != 0 and options.device == "auto":
                logger.warning(
                    "Whisper.cpp auto device invocation failed; retrying on CPU"
                )
                command = self._command(
                    audio, output_base, options, force_cpu=True
                )
                completed = self._run(command, options)
            if completed.returncode != 0:
                raise TranscriptionFailed(
                    "Whisper.cpp transcription",
                    command,
                    completed.returncode,
                    _diagnostic_output(completed),
                )

            output_path = output_base.with_suffix(".json")
            if not output_path.is_file():
                raise InvalidTranscriptOutput(
                    f"Whisper.cpp did not create JSON output: {output_path}"
                )
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise InvalidTranscriptOutput(
                    f"Could not read Whisper.cpp JSON output: {exc}"
                ) from exc
            return parse_whisper_output(payload, audio)

    def _command(
        self,
        audio: Path,
        output_base: Path,
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
            "-oj",
            "-ojf",
            "-of",
            str(output_base),
            "-l",
            options.language,
            "-t",
            str(options.threads),
            "-np",
        ]
        if options.translate:
            command.append("-tr")
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
                f"Whisper.cpp executable was not found: {self.executable}"
            ) from exc
        except OSError as exc:
            raise ExecutableNotFound(
                f"Could not start Whisper.cpp '{self.executable}': {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionFailed(
                "Whisper.cpp transcription timed out",
                command,
                -1,
                str(exc),
            ) from exc


def parse_whisper_output(
    payload: str | Mapping[str, Any], audio_path: PathLike
) -> Transcript:
    """Normalize whisper-cli JSON into the application transcript model."""

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InvalidTranscriptOutput(
                f"Invalid Whisper.cpp JSON for {audio_path}: {exc.msg}"
            ) from exc
    else:
        data = payload
    if not isinstance(data, Mapping):
        raise InvalidTranscriptOutput("Whisper.cpp JSON root must be an object")

    result = data.get("result", {})
    transcription = data.get("transcription", [])
    if not isinstance(result, Mapping) or not isinstance(transcription, list):
        raise InvalidTranscriptOutput(
            "Whisper.cpp JSON must contain result and transcription"
        )

    segments: list[SubtitleSegment] = []
    for index, raw_segment in enumerate(transcription, start=1):
        if not isinstance(raw_segment, Mapping):
            raise InvalidTranscriptOutput("Whisper.cpp segment must be an object")
        start_ms, end_ms = _segment_offsets(raw_segment)
        if end_ms < start_ms:
            raise InvalidTranscriptOutput(
                f"Whisper.cpp segment {index} ends before it starts"
            )
        text = str(raw_segment.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            SubtitleSegment(
                id=f"seg-{index:04d}",
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                words=_parse_words(raw_segment.get("tokens", [])),
            )
        )

    duration_ms = max((segment.end_ms for segment in segments), default=0)
    language = result.get("language")
    return Transcript(
        language=str(language) if language is not None else None,
        duration_ms=duration_ms,
        segments=segments,
    )


def _segment_offsets(segment: Mapping[str, Any]) -> tuple[int, int]:
    offsets = segment.get("offsets")
    if isinstance(offsets, Mapping):
        start = _int_value(offsets.get("from"))
        end = _int_value(offsets.get("to"))
        if start is not None and end is not None:
            return start, end
    timestamps = segment.get("timestamps")
    if isinstance(timestamps, Mapping):
        start = _timestamp_ms(timestamps.get("from"))
        end = _timestamp_ms(timestamps.get("to"))
        if start is not None and end is not None:
            return start, end
    raise InvalidTranscriptOutput("Whisper.cpp segment has no valid timestamps")


def _parse_words(tokens: Any) -> list[Word]:
    if not isinstance(tokens, list):
        return []
    words: list[Word] = []
    for token in tokens:
        if not isinstance(token, Mapping):
            continue
        text = str(token.get("text", "")).strip()
        if not text or _is_special_token(text):
            continue
        offsets = token.get("offsets")
        if not isinstance(offsets, Mapping):
            continue
        start = _int_value(offsets.get("from"))
        end = _int_value(offsets.get("to"))
        if start is None or end is None or end < start:
            continue
        confidence = token.get("p")
        try:
            parsed_confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            parsed_confidence = None
        words.append(Word(start, end, text, parsed_confidence))
    return words


def _is_special_token(text: str) -> bool:
    return text.startswith("[_") and text.endswith("]")


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timestamp_ms(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    parts = value.replace(".", ",").split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds, milliseconds = parts[2].split(",", 1)
        fraction = (milliseconds + "000")[:3]
        return ((hours * 60 + minutes) * 60 + int(seconds)) * 1000 + int(fraction)
    except (ValueError, IndexError):
        return None


def _diagnostic_output(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    return stderr or stdout or "no diagnostic output"
