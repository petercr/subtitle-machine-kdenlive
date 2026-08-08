import json
import subprocess
from pathlib import Path

import pytest

from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr import whisper_cpp as whisper_module
from video_mcp.asr.whisper_cpp import WhisperCppBackend, parse_whisper_output
from video_mcp.errors import ModelNotFound, TranscriptionFailed


def _whisper_payload() -> dict:
    return {
        "result": {"language": "en"},
        "transcription": [
            {
                "offsets": {"from": 0, "to": 1250},
                "text": " Hello, world!",
                "tokens": [
                    {
                        "text": "[_BEG_]",
                        "offsets": {"from": 0, "to": 0},
                        "p": 0.9,
                    },
                    {
                        "text": " Hello",
                        "offsets": {"from": 50, "to": 600},
                        "p": 0.98,
                    },
                    {
                        "text": ",",
                        "offsets": {"from": 600, "to": 700},
                        "p": 0.8,
                    },
                    {
                        "text": " world",
                        "offsets": {"from": 700, "to": 1100},
                        "p": 0.95,
                    },
                    {
                        "text": "!",
                        "offsets": {"from": 1100, "to": 1250},
                        "p": 0.7,
                    },
                    {
                        "text": "[_TT_10]",
                        "offsets": {"from": 1250, "to": 1250},
                        "p": 0.1,
                    },
                ],
            }
        ],
    }


def test_parse_whisper_output_creates_versioned_transcript():
    transcript = parse_whisper_output(_whisper_payload(), "audio.wav")

    assert transcript.schema_version == 1
    assert transcript.language == "en"
    assert transcript.duration_ms == 1250
    assert len(transcript.segments) == 1
    segment = transcript.segments[0]
    assert segment.id == "seg-0001"
    assert segment.text == "Hello, world!"
    assert [word.text for word in segment.words] == ["Hello", ",", "world", "!"]
    assert segment.words[0].confidence == pytest.approx(0.98)


def test_whisper_backend_uses_safe_arguments_for_spaced_paths(monkeypatch, tmp_path):
    audio_path = tmp_path / "Source Audio" / "Test Audio.wav"
    audio_path.parent.mkdir()
    audio_path.write_bytes(b"audio")
    executable = tmp_path / "Whisper Tools" / "whisper-cli.exe"
    model = tmp_path / "Whisper Models" / "ggml-small.bin"
    model.parent.mkdir()
    model.write_bytes(b"model")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_base = Path(command[command.index("-of") + 1])
        output_base.with_suffix(".json").write_text(
            json.dumps(_whisper_payload()), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(whisper_module.subprocess, "run", fake_run)

    transcript = WhisperCppBackend(executable, model).transcribe(
        audio_path,
        TranscriptionOptions(device="cpu", threads=6),
    )

    command, kwargs = calls[0]
    assert transcript.segments[0].text == "Hello, world!"
    assert command[0] == str(executable)
    assert command[command.index("-m") + 1] == str(model)
    assert command[command.index("-f") + 1] == str(audio_path)
    assert "-ng" in command
    assert command[command.index("-t") + 1] == "6"
    assert kwargs["shell"] is False


def test_whisper_auto_retries_on_cpu_after_failure(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    model = tmp_path / "model.bin"
    audio_path.write_bytes(b"audio")
    model.write_bytes(b"model")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="GPU failed")
        output_base = Path(command[command.index("-of") + 1])
        output_base.with_suffix(".json").write_text(
            json.dumps(_whisper_payload()), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(whisper_module.subprocess, "run", fake_run)

    transcript = WhisperCppBackend("whisper-cli", model).transcribe(
        audio_path, TranscriptionOptions(device="auto")
    )

    assert transcript.language == "en"
    assert len(calls) == 2
    assert "-ng" not in calls[0]
    assert "-ng" in calls[1]


def test_whisper_backend_requires_model(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    with pytest.raises(ModelNotFound, match="does not exist"):
        WhisperCppBackend("whisper-cli", tmp_path / "missing.bin").transcribe(audio_path)


def test_whisper_backend_reports_command_failure(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    model = tmp_path / "model.bin"
    audio_path.write_bytes(b"audio")
    model.write_bytes(b"model")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad audio")

    monkeypatch.setattr(whisper_module.subprocess, "run", fake_run)

    with pytest.raises(TranscriptionFailed, match="bad audio"):
        WhisperCppBackend("whisper-cli", model).transcribe(
            audio_path, TranscriptionOptions(device="cpu")
        )
