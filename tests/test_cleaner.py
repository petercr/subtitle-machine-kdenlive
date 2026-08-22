import json
import subprocess

import pytest

from video_mcp.config import AppConfig, ASRConfig, LLMConfig, OutputConfig, ToolConfig
from video_mcp.errors import InvalidCleanupOutput
from video_mcp.models import SubtitleSegment, Transcript
from video_mcp.services.cleanup import clean_transcript
from video_mcp.subtitles import cleaner as cleaner_module
from video_mcp.subtitles.cleaner import (
    DeterministicCleaner,
    LocalLLMCleaner,
    LocalLLMServerCleaner,
)


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_ms=3000,
        segments=[
            SubtitleSegment("seg-1", 0, 1000, " hello ,   world", []),
            SubtitleSegment("seg-2", 1000, 2000, "this is fine", []),
            SubtitleSegment("seg-3", 2000, 3000, "another line", []),
        ],
    )


def test_deterministic_cleaner_preserves_timing_and_normalizes_text():
    cleaned = DeterministicCleaner().clean(_transcript())

    assert [segment.text for segment in cleaned.segments] == [
        "Hello, world",
        "This is fine",
        "Another line",
    ]
    assert [(segment.start_ms, segment.end_ms) for segment in cleaned.segments] == [
        (0, 1000),
        (1000, 2000),
        (2000, 3000),
    ]


def test_local_llm_cleaner_chunks_and_validates_schema(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        prompt = command[command.index("--prompt") + 1]
        chunk = json.loads(prompt.split("Input transcript chunk:\n", 1)[1])
        ids = [item["id"] for item in chunk["segments"]]
        payload = {
            "segments": [{"id": segment_id, "text": f"Cleaned {segment_id}"} for segment_id in ids]
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=f"answer:\n{json.dumps(payload)}", stderr=""
        )

    monkeypatch.setattr(cleaner_module.subprocess, "run", fake_run)
    cleaned = LocalLLMCleaner(
        "llama-cli",
        model,
        max_segments_per_chunk=2,
    ).clean(_transcript())

    assert [segment.text for segment in cleaned.segments] == [
        "Cleaned seg-1",
        "Cleaned seg-2",
        "Cleaned seg-3",
    ]
    assert len(calls) == 2
    assert "--json-schema" in calls[0]
    assert calls[0][calls[0].index("--reasoning") + 1] == "off"
    assert calls[0][calls[0].index("--temperature") + 1] == "0"


def test_local_llm_cleaner_rejects_unknown_segment_ids(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"segments": [{"id": "wrong", "text": "bad"}]}),
            stderr="",
        )

    monkeypatch.setattr(cleaner_module.subprocess, "run", fake_run)

    with pytest.raises(InvalidCleanupOutput, match="unknown segment id"):
        LocalLLMCleaner("llama-cli", model).clean(_transcript())


def test_local_llm_server_cleaner_uses_strict_json_schema(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "segments": [
                                        {"id": "seg-1", "text": "Hello, world."},
                                        {"id": "seg-2", "text": "This is fine."},
                                        {"id": "seg-3", "text": "Another line."},
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request, *, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(cleaner_module, "urlopen", fake_urlopen)
    cleaned = LocalLLMServerCleaner("http://127.0.0.1:8087/v1/chat/completions").clean(
        _transcript()
    )

    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert timeout == 120
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert [segment.text for segment in cleaned.segments] == [
        "Hello, world.",
        "This is fine.",
        "Another line.",
    ]


def test_cleanup_service_falls_back_when_llm_is_unavailable(tmp_path):
    config = AppConfig(
        tools=ToolConfig(llama_cpp=tmp_path / "missing-llama.exe"),
        asr=ASRConfig(model=tmp_path / "model.bin"),
        llm=LLMConfig(enabled=True, model=tmp_path / "missing-model.gguf"),
        output=OutputConfig(workspace=tmp_path / "work"),
    )

    result = clean_transcript(_transcript(), config)

    assert result.used_llm is False
    assert result.transcript.segments[0].text == "Hello, world"
    assert result.warnings
