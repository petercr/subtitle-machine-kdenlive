import json
import subprocess
from pathlib import Path

import pytest

from video_mcp.errors import AudioExtractionFailed, MediaProbeFailed, UnsupportedMedia
from video_mcp.media import ffmpeg as ffmpeg_module
from video_mcp.media import probe as probe_module
from video_mcp.media.ffmpeg import extract_audio
from video_mcp.media.probe import parse_ffprobe_output, probe_video


def _ffprobe_payload() -> dict:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "tags": {"rotate": "90"},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "12.345",
            "size": "987654",
            "bit_rate": "640000",
        },
    }


def test_parse_ffprobe_output_normalizes_video_and_audio_metadata(tmp_path):
    input_path = tmp_path / "Videos" / "Test Video.mp4"

    info = parse_ffprobe_output(_ffprobe_payload(), input_path)

    assert info.path == input_path.resolve()
    assert info.duration_ms == 12345
    assert info.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
    assert info.size_bytes == 987654
    assert info.video is not None
    assert info.video.width == 1920
    assert info.video.height == 1080
    assert info.video.frame_rate == pytest.approx(29.97002997)
    assert info.video.rotation == 90
    assert info.audio is not None
    assert info.audio.sample_rate == 48000
    assert info.audio.channels == 2


def test_probe_video_passes_spaced_paths_as_single_arguments(monkeypatch, tmp_path):
    input_path = tmp_path / "Source Videos" / "Test Video.mp4"
    input_path.parent.mkdir()
    input_path.write_bytes(b"not a real video")
    ffprobe_path = tmp_path / "Tools With Spaces" / "ffprobe.exe"
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(_ffprobe_payload()), stderr=""
        )

    monkeypatch.setattr(probe_module.subprocess, "run", fake_run)

    info = probe_video(input_path, ffprobe_path=ffprobe_path)

    assert info.video is not None
    assert calls["command"][0] == str(ffprobe_path)
    assert calls["command"][-1] == str(input_path)
    assert calls["kwargs"]["shell"] is False
    assert calls["kwargs"]["capture_output"] is True


def test_probe_video_reports_ffprobe_failures(monkeypatch, tmp_path):
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"input")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad input")

    monkeypatch.setattr(probe_module.subprocess, "run", fake_run)

    with pytest.raises(MediaProbeFailed, match="bad input"):
        probe_video(input_path)


def test_parse_ffprobe_output_requires_a_video_stream(tmp_path):
    payload = {"streams": [{"codec_type": "audio"}], "format": {}}

    with pytest.raises(UnsupportedMedia, match="No video stream"):
        parse_ffprobe_output(payload, tmp_path / "audio.wav")


def test_extract_audio_builds_safe_mono_wav_command(monkeypatch, tmp_path):
    input_path = tmp_path / "Source Videos" / "Test Video.mp4"
    input_path.parent.mkdir()
    input_path.write_bytes(b"input")
    output_path = tmp_path / "Work Files" / "Test Video.wav"
    ffmpeg_path = tmp_path / "Tools With Spaces" / "ffmpeg.exe"
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        Path(command[-1]).write_bytes(b"RIFF" + b"\x00" * 8)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)

    result = extract_audio(input_path, output_path, ffmpeg_path=ffmpeg_path)

    command = calls["command"]
    assert result == output_path.resolve()
    assert command[0] == str(ffmpeg_path)
    assert command[command.index("-i") + 1] == str(input_path)
    assert command[-1] == str(output_path)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert calls["kwargs"]["shell"] is False
    assert result.is_file()


def test_extract_audio_does_not_overwrite_existing_output(tmp_path):
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"input")
    output_path = tmp_path / "audio.wav"
    output_path.write_bytes(b"original")

    with pytest.raises(AudioExtractionFailed, match="already exists"):
        extract_audio(input_path, output_path)

    assert output_path.read_bytes() == b"original"
