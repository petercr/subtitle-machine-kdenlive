import wave
from pathlib import Path

from video_mcp import evaluation as evaluation_module
from video_mcp.config import AppConfig, ASRConfig, OutputConfig, ToolConfig
from video_mcp.evaluation import audio_duration_seconds, benchmark_asr, word_error_rate
from video_mcp.models import SubtitleSegment, Transcript


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        tools=ToolConfig(),
        asr=ASRConfig(model=tmp_path / "model.bin"),
        output=OutputConfig(workspace=tmp_path / "work"),
    )


def test_word_error_rate_ignores_case_and_punctuation():
    assert word_error_rate("Hello, WORLD!", "hello world") == 0.0
    assert word_error_rate("one two three", "one four") == 2 / 3


def test_audio_duration_reads_normalized_wav(tmp_path):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)

    assert audio_duration_seconds(audio) == 1.0


def test_benchmark_asr_records_timing_and_wer(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)

    class FakeBackend:
        def transcribe(self, audio_path, options):
            return Transcript(
                language="en",
                duration_ms=1000,
                segments=[SubtitleSegment("seg-1", 0, 1000, "Hello world", [])],
            )

    monkeypatch.setattr(evaluation_module, "create_asr_backend", lambda config: FakeBackend())
    result = benchmark_asr(audio, _config(tmp_path), reference_text="hello world")

    assert result.error is None
    assert result.device == "auto"
    assert result.audio_duration_seconds == 1.0
    assert result.segment_count == 1
    assert result.word_error_rate == 0.0
    assert result.real_time_factor is not None
    assert result.peak_process_memory_mib is None
    assert result.peak_gpu_memory_mib is None
    assert result.gpu_memory_scope is None
    assert result.as_dict()["input_path"] == str(audio.resolve())


def test_gpu_memory_parser_counts_only_watched_processes(monkeypatch):
    class Completed:
        returncode = 0
        stdout = "123, 256\n456, 128\n"

    monkeypatch.setattr(evaluation_module.subprocess, "run", lambda *args, **kwargs: Completed())

    assert evaluation_module._gpu_memory_mib((123,)) == 256.0
    assert evaluation_module._gpu_memory_mib((789,)) is None


def test_gpu_device_memory_parser_sums_detected_devices(monkeypatch):
    class Completed:
        returncode = 0
        stdout = "512\n768\n"

    monkeypatch.setattr(evaluation_module.subprocess, "run", lambda *args, **kwargs: Completed())

    assert evaluation_module._gpu_device_memory_mib() == 1280.0
