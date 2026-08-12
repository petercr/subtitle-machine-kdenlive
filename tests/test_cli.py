import json

from video_mcp.cli import main


def test_cli_prints_help_without_command(capsys):
    assert main([]) == 0

    captured = capsys.readouterr()
    assert "Local-first video transcription" in captured.out


def test_cli_prints_effective_config(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert main(["config"]) == 0

    captured = capsys.readouterr()
    assert '"backend": "whisper_cpp"' in captured.out
    assert '"device": "auto"' in captured.out


def test_cli_cleans_transcript_with_deterministic_fallback(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    transcript_path = tmp_path / "raw.json"
    output_path = tmp_path / "cleaned.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "language": "en",
                "duration_ms": 1000,
                "segments": [
                    {
                        "id": "seg-1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": " hello ,   cli",
                        "words": [],
                        "speaker": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(
        ["clean", str(transcript_path), "--output", str(output_path), "--json"]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["used_llm"] is False
    assert result["segment_count"] == 1
    cleaned = json.loads(output_path.read_text(encoding="utf-8"))
    assert cleaned["segments"][0]["text"] == "Hello, cli"
