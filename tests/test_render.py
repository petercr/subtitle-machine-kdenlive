import subprocess
from pathlib import Path

import pytest

from video_mcp.errors import InputFileNotFound, RenderFailed
from video_mcp.media import render as render_module
from video_mcp.media.render import create_preview, render_subtitles


def test_create_preview_builds_safe_filter_and_output_command(monkeypatch, tmp_path):
    input_path = tmp_path / "Source Videos" / "Test Video.mp4"
    subtitles_path = tmp_path / "Subtitle Files" / "Test Video.ass"
    output_path = tmp_path / "Preview Files" / "Test Video Preview.mp4"
    ffmpeg_path = tmp_path / "FFmpeg Tools" / "ffmpeg.exe"
    input_path.parent.mkdir()
    subtitles_path.parent.mkdir()
    input_path.write_bytes(b"video")
    subtitles_path.write_text("[Script Info]\n", encoding="utf-8")
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(render_module.subprocess, "run", fake_run)

    result = create_preview(
        input_path,
        subtitles_path,
        output_path,
        ffmpeg_path=ffmpeg_path,
    )

    command = calls["command"]
    filter_value = command[command.index("-vf") + 1]
    assert result == output_path.resolve()
    assert command[0] == str(ffmpeg_path)
    assert command[command.index("-i") + 1] == str(input_path)
    assert "subtitles=filename='" in filter_value
    assert "Test Video.ass" in filter_value
    assert "scale=1280:-2" in filter_value
    assert "-preset" in command and command[command.index("-preset") + 1] == "veryfast"
    assert calls["kwargs"]["shell"] is False


def test_render_subtitles_uses_full_resolution_encoding_when_not_preview(monkeypatch, tmp_path):
    input_path = tmp_path / "video.mp4"
    subtitles_path = tmp_path / "captions.ass"
    output_path = tmp_path / "rendered.mp4"
    input_path.write_bytes(b"video")
    subtitles_path.write_text("[Script Info]\n", encoding="utf-8")
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(render_module.subprocess, "run", fake_run)

    render_subtitles(input_path, subtitles_path, output_path, preview=False)

    command = calls["command"]
    filter_value = command[command.index("-vf") + 1]
    assert "scale=" not in filter_value
    assert command[command.index("-preset") + 1] == "medium"
    assert command[command.index("-crf") + 1] == "20"


def test_render_subtitles_reports_ffmpeg_failure(monkeypatch, tmp_path):
    input_path = tmp_path / "video.mp4"
    subtitles_path = tmp_path / "captions.ass"
    output_path = tmp_path / "rendered.mp4"
    input_path.write_bytes(b"video")
    subtitles_path.write_text("[Script Info]\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad filter")

    monkeypatch.setattr(render_module.subprocess, "run", fake_run)

    with pytest.raises(RenderFailed, match="bad filter"):
        render_subtitles(input_path, subtitles_path, output_path)


def test_render_requires_existing_input_and_subtitles(tmp_path):
    subtitles_path = tmp_path / "captions.ass"
    subtitles_path.write_text("[Script Info]\n", encoding="utf-8")

    with pytest.raises(InputFileNotFound, match="Input video"):
        render_subtitles(tmp_path / "missing.mp4", subtitles_path, tmp_path / "out.mp4")
