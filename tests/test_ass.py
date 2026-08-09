import pytest

from video_mcp.errors import SubtitleGenerationFailed
from video_mcp.models import SubtitleSegment, Transcript
from video_mcp.subtitles.ass import (
    escape_ass_text,
    format_ass_timestamp,
    generate_ass,
    write_ass,
)
from video_mcp.subtitles.styles import ASSStyle, get_style_preset


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_ms=3000,
        segments=[
            SubtitleSegment(
                id="segment-1",
                start_ms=0,
                end_ms=1255,
                text="Hello {world}\nUse a \\ path.",
                words=[],
            )
        ],
    )


@pytest.mark.parametrize(
    ("milliseconds", "expected"),
    [
        (0, "0:00:00.00"),
        (1250, "0:00:01.25"),
        (1255, "0:00:01.26"),
        (3723004, "1:02:03.00"),
    ],
)
def test_format_ass_timestamp(milliseconds, expected):
    assert format_ass_timestamp(milliseconds) == expected


def test_ass_timestamp_rejects_negative_values():
    with pytest.raises(SubtitleGenerationFailed, match="negative"):
        format_ass_timestamp(-1)


def test_escape_ass_text_protects_tags_and_line_breaks():
    assert escape_ass_text("a {tag}\r\nb \\ path") == "a \\{tag\\}\\Nb \\\\ path"


def test_generate_ass_includes_clean_style_and_escaped_dialogue():
    output = generate_ass(_transcript(), play_res_x=1280, play_res_y=720)

    assert "PlayResX: 1280" in output
    assert "PlayResY: 720" in output
    assert "Style: clean,Arial,48" in output
    assert "Dialogue: 0,0:00:00.00,0:00:01.26,clean,,0,0,0,," in output
    assert "Hello \\{world\\}\\NUse a \\\\ path." in output
    assert output.endswith("\n")


def test_custom_style_can_emit_bold_ass_style():
    style = ASSStyle(name="custom", bold=True, font_size=64)

    output = generate_ass(_transcript(), style)

    assert "Style: custom,Arial,64" in output
    assert ",&H80000000,-1,0,0,0,100,100," in output


def test_ass_rejects_unknown_style_and_overlapping_segments():
    with pytest.raises(SubtitleGenerationFailed, match="Unknown ASS style"):
        get_style_preset("missing")

    overlapping = Transcript(
        language="en",
        duration_ms=3000,
        segments=[
            SubtitleSegment("one", 0, 2000, "one", []),
            SubtitleSegment("two", 1500, 3000, "two", []),
        ],
    )
    with pytest.raises(SubtitleGenerationFailed, match="overlaps"):
        generate_ass(overlapping)


def test_write_ass_supports_spaced_paths_and_preserves_existing_output(tmp_path):
    output = tmp_path / "Subtitle Files" / "Test Video.ass"
    result = write_ass(_transcript(), output)

    assert result == output.resolve()
    assert output.read_text(encoding="utf-8").startswith("[Script Info]\n")
    with pytest.raises(SubtitleGenerationFailed, match="already exists"):
        write_ass(_transcript(), output)
