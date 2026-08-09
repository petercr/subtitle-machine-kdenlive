import subprocess
from pathlib import Path

import pytest

from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr.factory import create_asr_backend
from video_mcp.asr import parakeet as parakeet_module
from video_mcp.asr.parakeet import ParakeetBackend, parse_parakeet_output
from video_mcp.config import AppConfig, ASRConfig, ToolConfig
from video_mcp.errors import InvalidTranscriptOutput, ModelNotFound, TranscriptionFailed


PARAKEET_OUTPUT = """
Processing audio (176000 samples, 11.00 seconds)
Segments (1):
Segment 0: [0 -> 1101] "And so, my fellow Americans."
Tokens [4]:
 [ 0] id= 1976 frame= 3 dur_idx= 4 dur_val= 4 p=0.9996 plog=-15.6206 t0= 24 t1= 56 word_start=true "▁And"
 [ 1] id= 547 frame= 7 dur_idx= 4 dur_val= 4 p=0.9999 plog=-18.7922 t0= 56 t1= 88 word_start=true "▁so"
 [ 2] id= 7877 frame= 11 dur_idx= 2 dur_val= 2 p=0.8451 plog=-14.5929 t0= 88 t1= 88 word_start=false ","
 [ 3] id= 1103 frame= 13 dur_idx= 3 dur_val= 3 p=0.9996 plog=-15.6127 t0= 104 t1= 128 word_start=true "▁my"
"""


def test_parse_parakeet_output_normalizes_frames_and_words():
    transcript = parse_parakeet_output(PARAKEET_OUTPUT, "audio.wav")

    assert transcript.language is None
    assert transcript.duration_ms == 11010
    assert transcript.segments[0].text == "And so, my fellow Americans."
    assert [(word.text, word.start_ms, word.end_ms) for word in transcript.segments[0].words] == [
        ("And", 240, 560),
        ("so,", 560, 880),
        ("my", 1040, 1280),
    ]


def test_parse_parakeet_output_requires_segments():
    with pytest.raises(InvalidTranscriptOutput, match="no segment output"):
        parse_parakeet_output("Loading Parakeet model", "audio.wav")


def test_factory_selects_parakeet_backend(tmp_path):
    config = AppConfig(
        tools=ToolConfig(parakeet=tmp_path / "parakeet-cli.exe"),
        asr=ASRConfig(backend="parakeet", model=tmp_path / "model.bin"),
    )

    backend = create_asr_backend(config)

    assert isinstance(backend, ParakeetBackend)


def test_parakeet_backend_uses_safe_arguments_and_parses_stderr(monkeypatch, tmp_path):
    audio = tmp_path / "Source Audio.wav"
    model = tmp_path / "parakeet q8.bin"
    audio.write_bytes(b"audio")
    model.write_bytes(b"model")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="", stderr=PARAKEET_OUTPUT
        )

    monkeypatch.setattr(parakeet_module.subprocess, "run", fake_run)

    transcript = ParakeetBackend("parakeet-cli.exe", model).transcribe(
        audio, TranscriptionOptions(device="cpu", threads=6)
    )

    assert transcript.segments[0].text.startswith("And so")
    assert calls[0][0] == "parakeet-cli.exe"
    assert calls[0][calls[0].index("-m") + 1] == str(model)
    assert calls[0][calls[0].index("-f") + 1] == str(audio)
    assert "-ps" in calls[0]
    assert "-np" in calls[0]
    assert "-ng" in calls[0]


def test_parakeet_auto_retries_on_cpu_after_failure(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    model = tmp_path / "model.bin"
    audio.write_bytes(b"audio")
    model.write_bytes(b"model")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="GPU failed")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=PARAKEET_OUTPUT)

    monkeypatch.setattr(parakeet_module.subprocess, "run", fake_run)

    transcript = ParakeetBackend("parakeet-cli", model).transcribe(
        audio, TranscriptionOptions(device="auto")
    )

    assert transcript.segments
    assert "-ng" not in calls[0]
    assert "-ng" in calls[1]


def test_parakeet_backend_requires_model(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    with pytest.raises(ModelNotFound, match="does not exist"):
        ParakeetBackend("parakeet-cli", tmp_path / "missing.bin").transcribe(audio)


def test_parakeet_backend_reports_command_failure(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    model = tmp_path / "model.bin"
    audio.write_bytes(b"audio")
    model.write_bytes(b"model")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad audio")

    monkeypatch.setattr(parakeet_module.subprocess, "run", fake_run)

    with pytest.raises(TranscriptionFailed, match="bad audio"):
        ParakeetBackend("parakeet-cli", model).transcribe(
            audio, TranscriptionOptions(device="cpu")
        )
