import json
from pathlib import Path

from video_mcp.config import AppConfig, ASRConfig, OutputConfig, ToolConfig
from video_mcp.models import (
    AudioStreamInfo,
    MediaInfo,
    SubtitleSegment,
    Transcript,
    VideoStreamInfo,
)
from video_mcp.services import captioning as captioning_module
from video_mcp.services.captioning import CaptionOptions, caption_video


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        tools=ToolConfig(),
        asr=ASRConfig(model=tmp_path / "model.bin"),
        output=OutputConfig(workspace=tmp_path / "work"),
    )


def _media(input_path: Path) -> MediaInfo:
    return MediaInfo(
        path=input_path,
        duration_ms=2500,
        format_name="mp4",
        size_bytes=123,
        bit_rate=1000,
        video=VideoStreamInfo("h264", 1920, 1080, 30.0, None),
        audio=AudioStreamInfo("aac", 48000, 2, "stereo"),
    )


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_ms=2000,
        segments=[SubtitleSegment("seg-1", 0, 2000, "Hello pipeline", [])],
    )


def test_caption_video_creates_expected_job_artifacts(monkeypatch, tmp_path):
    source = tmp_path / "Source Videos" / "Test Video.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    config = _config(tmp_path)
    calls = {"extract": 0, "transcribe": 0, "preview": 0}
    log_events = []

    class FakeLogger:
        def info(self, message, **kwargs):
            log_events.append((message, kwargs.get("extra", {})))

    monkeypatch.setattr(captioning_module, "new_job_id", lambda: "job-123")
    monkeypatch.setattr(captioning_module, "get_job_logger", lambda *args, **kwargs: FakeLogger())

    monkeypatch.setattr(captioning_module, "probe_video", lambda path, **kwargs: _media(source))

    def fake_extract(input_path, output_path, **kwargs):
        calls["extract"] += 1
        Path(output_path).write_bytes(b"wav")
        return Path(output_path).resolve()

    monkeypatch.setattr(captioning_module, "extract_audio", fake_extract)

    class FakeBackend:
        def transcribe(self, audio_path, options):
            calls["transcribe"] += 1
            return _transcript()

    monkeypatch.setattr(captioning_module, "create_asr_backend", lambda config: FakeBackend())

    def fake_preview(input_path, subtitles_path, output_path, **kwargs):
        calls["preview"] += 1
        Path(output_path).write_bytes(b"preview")
        return Path(output_path).resolve()

    monkeypatch.setattr(captioning_module, "create_preview", fake_preview)

    result = caption_video(source, config, CaptionOptions(device="cpu"))

    assert result.job_id == "job-123"
    assert result.job_dir == (tmp_path / "work" / "Test Video").resolve()
    for path in (
        result.source_json,
        result.audio_wav,
        result.transcript_raw_json,
        result.transcript_cleaned_json,
        result.subtitles_srt,
        result.subtitles_ass,
        result.preview_mp4,
    ):
        assert path is not None and path.is_file()
    assert result.segment_count == 1
    assert result.warnings
    assert json.loads(result.transcript_raw_json.read_text()) == json.loads(
        result.transcript_cleaned_json.read_text()
    )
    assert calls == {"extract": 1, "transcribe": 1, "preview": 1}
    assert [event[0] for event in log_events] == [
        "Caption job started",
        "Media inspection completed",
        "Source metadata ready",
        "Audio extraction completed",
        "Transcription completed",
        "Transcript cleanup completed",
        "SRT export completed",
        "ASS export completed",
        "Preview rendering completed",
        "Caption job completed",
    ]


def test_caption_video_reuses_existing_artifacts(monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    config = _config(tmp_path)
    calls = {"extract": 0, "transcribe": 0, "preview": 0}

    monkeypatch.setattr(captioning_module, "probe_video", lambda path, **kwargs: _media(source))

    def fake_extract(input_path, output_path, **kwargs):
        calls["extract"] += 1
        Path(output_path).write_bytes(b"wav")
        return Path(output_path).resolve()

    monkeypatch.setattr(captioning_module, "extract_audio", fake_extract)

    class FakeBackend:
        def transcribe(self, audio_path, options):
            calls["transcribe"] += 1
            return _transcript()

    monkeypatch.setattr(captioning_module, "create_asr_backend", lambda config: FakeBackend())

    def fake_preview(input_path, subtitles_path, output_path, **kwargs):
        calls["preview"] += 1
        Path(output_path).write_bytes(b"preview")
        return Path(output_path).resolve()

    monkeypatch.setattr(captioning_module, "create_preview", fake_preview)

    caption_video(source, config, CaptionOptions(device="cpu"))
    caption_video(source, config, CaptionOptions(device="cpu"))

    assert calls == {"extract": 1, "transcribe": 1, "preview": 1}
