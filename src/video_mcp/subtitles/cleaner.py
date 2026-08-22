"""Deterministic and optional local-LLM transcript cleanup."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from video_mcp.errors import (
    CleanupFailed,
    ExecutableNotFound,
    InvalidCleanupOutput,
    ModelNotFound,
)
from video_mcp.models import SubtitleSegment, Transcript

logger = logging.getLogger(__name__)
PathLike = str | Path

CLEANUP_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["segments"],
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "text": {"type": "string"},
                },
            },
        }
    },
}


class SubtitleCleaner(Protocol):
    """Interface implemented by transcript cleanup strategies."""

    def clean(self, transcript: Transcript) -> Transcript:
        """Return a cleaned transcript without changing timing metadata."""


class DeterministicCleaner:
    """Apply safe whitespace and capitalization cleanup without a model."""

    def clean(self, transcript: Transcript) -> Transcript:
        return replace(
            transcript,
            segments=[
                replace(segment, text=_clean_text(segment.text)) for segment in transcript.segments
            ],
        )


class LocalLLMCleaner:
    """Use llama.cpp constrained JSON generation for conservative cleanup."""

    def __init__(
        self,
        executable: PathLike,
        model: PathLike,
        *,
        max_segments_per_chunk: int = 8,
        max_chars_per_chunk: int = 2400,
        max_tokens: int = 512,
        timeout_seconds: float = 120,
    ) -> None:
        if max_segments_per_chunk <= 0:
            raise ValueError("max_segments_per_chunk must be greater than zero")
        if max_chars_per_chunk <= 0:
            raise ValueError("max_chars_per_chunk must be greater than zero")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.executable = Path(executable).expanduser()
        self.model = Path(model).expanduser()
        self.max_segments_per_chunk = max_segments_per_chunk
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def clean(self, transcript: Transcript) -> Transcript:
        if not transcript.segments:
            return transcript

        cleaned_by_id: dict[str, str] = {}
        for chunk in self._chunks(transcript.segments):
            response = self._run_chunk(chunk)
            cleaned_by_id.update(_validate_cleanup_response(response, chunk))

        return replace(
            transcript,
            segments=[
                replace(segment, text=cleaned_by_id[segment.id]) for segment in transcript.segments
            ],
        )

    def _chunks(self, segments: list[SubtitleSegment]) -> list[list[SubtitleSegment]]:
        chunks: list[list[SubtitleSegment]] = []
        current: list[SubtitleSegment] = []
        current_chars = 0
        for segment in segments:
            segment_chars = len(segment.text)
            reaches_limit = current and (
                len(current) >= self.max_segments_per_chunk
                or current_chars + segment_chars > self.max_chars_per_chunk
            )
            if reaches_limit:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(segment)
            current_chars += segment_chars
        if current:
            chunks.append(current)
        return chunks

    def _run_chunk(self, segments: list[SubtitleSegment]) -> str:
        if not self.model.is_file():
            raise ModelNotFound(f"LLM cleanup model does not exist: {self.model}")

        prompt = _cleanup_prompt(segments)
        command = [
            str(self.executable),
            "-m",
            str(self.model),
            "--simple-io",
            "--no-display-prompt",
            "--no-show-timings",
            "--single-turn",
            "--reasoning",
            "off",
            "--temperature",
            "0",
            "--seed",
            "0",
            "--predict",
            str(self.max_tokens),
            "--json-schema",
            json.dumps(CLEANUP_RESPONSE_SCHEMA, separators=(",", ":")),
            "--prompt",
            prompt,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ExecutableNotFound(
                f"llama.cpp executable was not found: {self.executable}"
            ) from exc
        except OSError as exc:
            raise ExecutableNotFound(
                f"Could not start llama.cpp '{self.executable}': {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CleanupFailed("Local LLM cleanup timed out", command, -1, str(exc)) from exc

        if completed.returncode != 0:
            diagnostic = completed.stderr.strip() or completed.stdout.strip()
            raise CleanupFailed(
                "Local LLM cleanup",
                command,
                completed.returncode,
                diagnostic or "no diagnostic output",
            )
        return completed.stdout


class LocalLLMServerCleaner:
    """Use a running local llama.cpp OpenAI-compatible server for cleanup."""

    def __init__(
        self,
        endpoint: str,
        *,
        max_segments_per_chunk: int = 8,
        max_chars_per_chunk: int = 2400,
        max_tokens: int = 512,
        timeout_seconds: float = 120,
    ) -> None:
        self.endpoint = endpoint
        self.max_segments_per_chunk = max_segments_per_chunk
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def clean(self, transcript: Transcript) -> Transcript:
        if not transcript.segments:
            return transcript

        cleaned_by_id: dict[str, str] = {}
        for chunk in LocalLLMCleaner._chunks(self, transcript.segments):
            cleaned_by_id.update(_validate_cleanup_response(self._run_chunk(chunk), chunk))
        return replace(
            transcript,
            segments=[
                replace(segment, text=cleaned_by_id[segment.id]) for segment in transcript.segments
            ],
        )

    def _run_chunk(self, segments: list[SubtitleSegment]) -> str:
        payload = {
            "messages": [{"role": "user", "content": _cleanup_prompt(segments)}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "subtitle_cleanup",
                    "strict": True,
                    "schema": CLEANUP_RESPONSE_SCHEMA,
                },
            },
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise CleanupFailed(
                "Local llama.cpp server cleanup", [self.endpoint], -1, str(exc)
            ) from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidCleanupOutput(
                "Local llama.cpp server returned no message content"
            ) from exc
        if not isinstance(content, str):
            raise InvalidCleanupOutput("Local llama.cpp server returned non-text message content")
        return content


def _cleanup_prompt(segments: list[SubtitleSegment]) -> str:
    payload = {"segments": [{"id": segment.id, "text": segment.text} for segment in segments]}
    return (
        "Clean up ASR subtitle text conservatively. Correct punctuation, capitalization, "
        "obvious recognition errors, and sentence boundaries only. Do not summarize, "
        "paraphrase, add facts, remove meaning, change segment IDs, or merge/split segments. "
        "Return JSON only matching the requested schema.\n\n"
        f"Input transcript chunk:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _validate_cleanup_response(
    payload: str, expected_segments: list[SubtitleSegment]
) -> dict[str, str]:
    try:
        data = _decode_json(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InvalidCleanupOutput(f"Cleanup output was not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"segments"}:
        raise InvalidCleanupOutput("Cleanup output must contain only a segments array")
    raw_segments = data["segments"]
    if not isinstance(raw_segments, list):
        raise InvalidCleanupOutput("Cleanup segments must be an array")

    expected_ids = {segment.id for segment in expected_segments}
    cleaned: dict[str, str] = {}
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict) or set(raw_segment) != {"id", "text"}:
            raise InvalidCleanupOutput("Each cleanup segment must contain only id and text")
        segment_id = raw_segment["id"]
        text = raw_segment["text"]
        if not isinstance(segment_id, str) or not isinstance(text, str):
            raise InvalidCleanupOutput("Cleanup segment id and text must be strings")
        if segment_id not in expected_ids:
            raise InvalidCleanupOutput(f"Cleanup returned unknown segment id: {segment_id}")
        if segment_id in cleaned:
            raise InvalidCleanupOutput(f"Cleanup returned duplicate segment id: {segment_id}")
        cleaned[segment_id] = text.strip()
    if set(cleaned) != expected_ids:
        raise InvalidCleanupOutput("Cleanup must return exactly the input segment IDs")
    return cleaned


def _decode_json(payload: str) -> Any:
    stripped = payload.strip()
    if not stripped:
        raise ValueError("empty output")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise first_error


def _clean_text(text: str) -> str:
    cleaned = " ".join(text.split())
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    for index, character in enumerate(cleaned):
        if character.isalpha():
            cleaned = cleaned[:index] + character.upper() + cleaned[index + 1 :]
            break
    return cleaned
