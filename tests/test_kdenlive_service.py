import json
from pathlib import Path

import pytest

import video_mcp.cli as cli_module
from video_mcp.config import AppConfig, OutputConfig, ToolConfig
from video_mcp.errors import KdenliveProjectFailed
from video_mcp.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from video_mcp.services import kdenlive as kdenlive_service
from video_mcp.services.kdenlive import (
    KdenliveProjectResult,
    create_kdenlive_project,
)


def _media(source: Path) -> MediaInfo:
    return MediaInfo(
        path=source,
        duration_ms=2500,
        format_name="mp4",
        size_bytes=123,
        bit_rate=1000,
        video=VideoStreamInfo("h264", 1920, 1080, 30.0, None),
        audio=AudioStreamInfo("aac", 48000, 2, "stereo"),
    )


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        tools=ToolConfig(),
        output=OutputConfig(workspace=tmp_path / "work"),
    )


def test_create_kdenlive_project_uses_workspace_default_and_returns_paths(monkeypatch, tmp_path):
    source = tmp_path / "Source Videos" / "Test Video.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    subtitles = tmp_path / "subtitles.srt"
    subtitles.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    monkeypatch.setattr(kdenlive_service, "probe_video", lambda *args, **kwargs: _media(source))

    result = create_kdenlive_project(source, subtitles, _config(tmp_path))

    assert result.project_path == (tmp_path / "work" / "Test Video-captioned.kdenlive").resolve()
    assert result.project_subtitles_path == Path(f"{result.project_path}.srt")
    assert result.project_path.is_file()
    assert result.project_subtitles_path.is_file()
    assert result.duration_ms == 2500
    assert result.width == 1920
    assert result.height == 1080
    assert result.as_dict()["project_path"] == str(result.project_path)


def test_create_kdenlive_project_requires_srt_and_preserves_existing_output(monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    subtitles = tmp_path / "subtitles.ass"
    subtitles.write_text("[Script Info]\n", encoding="utf-8")
    config = _config(tmp_path)

    with pytest.raises(KdenliveProjectFailed, match=r"requires an \.srt file"):
        create_kdenlive_project(source, subtitles, config)

    subtitles = tmp_path / "subtitles.srt"
    subtitles.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    monkeypatch.setattr(kdenlive_service, "probe_video", lambda *args, **kwargs: _media(source))
    create_kdenlive_project(source, subtitles, config)

    with pytest.raises(KdenliveProjectFailed, match="Output already exists"):
        create_kdenlive_project(source, subtitles, config)


def test_cli_kdenlive_emits_structured_result(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "video.mp4"
    subtitles = tmp_path / "subtitles.srt"
    result = KdenliveProjectResult(
        input_path=source,
        subtitles_path=subtitles,
        project_path=tmp_path / "video-captioned.kdenlive",
        project_subtitles_path=tmp_path / "video-captioned.kdenlive.srt",
        duration_ms=2500,
        width=1920,
        height=1080,
        frame_rate=30.0,
    )
    monkeypatch.setattr(cli_module, "create_kdenlive_project", lambda *args, **kwargs: result)

    assert cli_module.main(["kdenlive", str(source), "--subtitles", str(subtitles), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["project_path"] == str(result.project_path)
    assert payload["project_subtitles_path"] == str(result.project_subtitles_path)
