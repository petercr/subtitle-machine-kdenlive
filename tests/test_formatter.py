from video_mcp.models import SubtitleSegment, Transcript, Word
from video_mcp.subtitles.formatter import _words_text, format_transcript


def test_formatter_splits_long_word_timestamp_segment_at_sentence_boundaries():
    words = [
        Word(0, 500, "One"),
        Word(500, 1000, "short"),
        Word(1000, 1500, "sentence."),
        Word(2000, 2500, "Another"),
        Word(2500, 3000, "sentence"),
        Word(3000, 3500, "follows."),
    ]
    transcript = Transcript(
        language="en",
        duration_ms=3500,
        segments=[
            SubtitleSegment(
                "seg-1",
                0,
                3500,
                "One short sentence. Another sentence follows.",
                words,
            )
        ],
    )

    formatted = format_transcript(transcript, max_chars=28)

    assert [(item.id, item.start_ms, item.end_ms, item.text) for item in formatted.segments] == [
        ("seg-1-001", 0, 1500, "One short sentence."),
        ("seg-1-002", 2000, 3500, "Another sentence follows."),
    ]
    assert all(
        previous.end_ms <= current.start_ms
        for previous, current in zip(formatted.segments, formatted.segments[1:])
    )


def test_formatter_wraps_split_cues_to_the_configured_line_length():
    words = [
        Word(index * 500, (index + 1) * 500, text)
        for index, text in enumerate(("This", "is", "a", "long", "caption", "line."))
    ]
    transcript = Transcript(
        language="en",
        duration_ms=3000,
        segments=[
            SubtitleSegment(
                "seg-1", 0, 3000, "This is a long caption line.", words
            )
        ],
    )

    formatted = format_transcript(
        transcript, max_chars=20, max_chars_per_line=12
    )

    assert formatted.segments[0].text == "This is a\nlong"


def test_formatter_leaves_compact_segments_unchanged():
    segment = SubtitleSegment("seg-1", 0, 1000, "Hello, world!", [])
    transcript = Transcript(language="en", duration_ms=1000, segments=[segment])

    assert format_transcript(transcript, max_chars=42).segments == [segment]


def test_formatter_attaches_contraction_suffixes_to_the_previous_word():
    assert _words_text([Word(0, 100, "That"), Word(100, 200, "'s")]) == "That's"
