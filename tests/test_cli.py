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
