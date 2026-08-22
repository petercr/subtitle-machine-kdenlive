"""Deterministic caption segmentation from ASR word timestamps."""

from __future__ import annotations

from dataclasses import replace

from video_mcp.models import SubtitleSegment, Transcript, Word

_TERMINAL_PUNCTUATION = (".", "!", "?")
_PHRASE_PUNCTUATION = (",", ";", ":")
_NO_SPACE_BEFORE = set(",.;:!?%)]}'’")


def format_transcript(
    transcript: Transcript,
    *,
    max_chars: int,
    max_duration_ms: int = 6000,
    max_chars_per_line: int | None = None,
) -> Transcript:
    """Split oversized timestamped ASR segments into readable caption cues.

    Backends such as Parakeet can return one accurate, long segment with word
    timings. This formatter preserves those word timings, favours sentence and
    phrase boundaries, and never changes compact existing segments.
    """

    if max_chars <= 0 or max_duration_ms <= 0:
        raise ValueError("caption limits must be greater than zero")

    formatted: list[SubtitleSegment] = []
    for segment in transcript.segments:
        if (
            not segment.words
            or len(segment.text) <= max_chars
            and segment.end_ms - segment.start_ms <= max_duration_ms
        ):
            formatted.append(segment)
            continue
        formatted.extend(_split_segment(segment, max_chars, max_duration_ms, max_chars_per_line))
    return replace(transcript, segments=formatted)


def _split_segment(
    segment: SubtitleSegment,
    max_chars: int,
    max_duration_ms: int,
    max_chars_per_line: int | None,
) -> list[SubtitleSegment]:
    words = segment.words
    pieces: list[SubtitleSegment] = []
    start = 0
    piece_number = 1
    while start < len(words):
        limit = start + 1
        while limit < len(words):
            candidate = words[start : limit + 1]
            if (
                len(_words_text(candidate)) > max_chars
                or candidate[-1].end_ms - candidate[0].start_ms > max_duration_ms
            ):
                break
            limit += 1
        endpoint = _preferred_endpoint(words, start, limit)
        cue_words = words[start:endpoint]
        pieces.append(
            SubtitleSegment(
                id=f"{segment.id}-{piece_number:03d}",
                start_ms=cue_words[0].start_ms,
                end_ms=cue_words[-1].end_ms,
                text=_wrap_text(_words_text(cue_words), max_chars_per_line),
                words=cue_words,
                speaker=segment.speaker,
            )
        )
        start = endpoint
        piece_number += 1
    return pieces


def _preferred_endpoint(words: list[Word], start: int, limit: int) -> int:
    """Choose the latest natural boundary without exceeding the hard limit."""

    if len(words) - limit == 1:
        return len(words)
    for punctuation in (_TERMINAL_PUNCTUATION, _PHRASE_PUNCTUATION):
        for index in range(limit - 1, start, -1):
            if words[index].text.rstrip().endswith(punctuation):
                return index + 1
    return limit


def _words_text(words: list[Word]) -> str:
    text = ""
    for word in words:
        value = word.text.strip()
        if not value:
            continue
        if text and value[0] not in _NO_SPACE_BEFORE:
            text += " "
        text += value
    return text


def _wrap_text(text: str, max_chars_per_line: int | None) -> str:
    if max_chars_per_line is None or len(text) <= max_chars_per_line:
        return text
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > max_chars_per_line:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)
