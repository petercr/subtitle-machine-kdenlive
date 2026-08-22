import json

import pytest

from video_mcp.errors import SubtitleGenerationFailed
from video_mcp.models import SubtitleSegment, Transcript, Word
from video_mcp.subtitles.srt import format_srt_timestamp, generate_srt, write_srt


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_ms=4000,
        segments=[
            SubtitleSegment(
                id="seg-a",
                start_ms=0,
                end_ms=1250,
                text="Hello, world!",
                words=[Word(50, 600, "Hello", 0.98)],
            ),
            SubtitleSegment(
                id="seg-b",
                start_ms=1500,
                end_ms=3723004,
                text="This is a second cue.\r\nWith two lines.",
                words=[],
            ),
        ],
    )


@pytest.mark.parametrize(
    ("milliseconds", "expected"),
    [
        (0, "00:00:00,000"),
        (1250, "00:00:01,250"),
        (3723004, "01:02:03,004"),
    ],
)
def test_format_srt_timestamp(milliseconds, expected):
    assert format_srt_timestamp(milliseconds) == expected


def test_format_srt_timestamp_rejects_negative_values():
    with pytest.raises(SubtitleGenerationFailed, match="negative"):
        format_srt_timestamp(-1)


def test_generate_srt_is_deterministic_and_normalizes_line_endings():
    assert generate_srt(_transcript()) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "Hello, world!\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 01:02:03,004\n"
        "This is a second cue.\n"
        "With two lines.\n"
    )


def test_generate_srt_rejects_overlap_and_invalid_duration():
    overlapping = Transcript(
        language="en",
        duration_ms=3000,
        segments=[
            SubtitleSegment("one", 0, 2000, "one", []),
            SubtitleSegment("two", 1500, 3000, "two", []),
        ],
    )
    with pytest.raises(SubtitleGenerationFailed, match="overlaps"):
        generate_srt(overlapping)

    invalid_duration = Transcript(
        language="en",
        duration_ms=0,
        segments=[SubtitleSegment("one", 2, 2, "one", [])],
    )
    with pytest.raises(SubtitleGenerationFailed, match="end after"):
        generate_srt(invalid_duration)


def test_write_srt_supports_spaced_paths_and_preserves_existing_output(tmp_path):
    output = tmp_path / "Subtitle Files" / "Test Video.srt"
    result = write_srt(_transcript(), output)

    assert result == output.resolve()
    assert output.read_text(encoding="utf-8").startswith("1\n00:00:00,000")
    with pytest.raises(SubtitleGenerationFailed, match="already exists"):
        write_srt(_transcript(), output)


def test_transcript_json_round_trip():
    original = _transcript()

    restored = Transcript.from_dict(json.loads(json.dumps(original.as_dict())))

    assert restored == original


def test_transcript_from_dict_rejects_unknown_schema():
    with pytest.raises(ValueError, match="Unsupported transcript schema"):
        Transcript.from_dict({"schema_version": 99, "segments": []})
