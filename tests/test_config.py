from pathlib import Path

import pytest

from video_mcp.config import ConfigurationError, load_config


def test_defaults_are_resolved_from_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_config(environ={})

    assert config.tools.ffmpeg == Path("ffmpeg")
    assert config.asr.device == "auto"
    assert config.asr.model == Path("C:/Models/whisper/ggml-small.bin")
    assert config.output.workspace == tmp_path / "work"
    assert config.source_path is None


def test_yaml_paths_resolve_relative_to_configuration(tmp_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text(
        """
tools:
  ffmpeg: ./tools/ffmpeg.exe
asr:
  device: cpu
  model: ./models/model.bin
output:
  workspace: ./generated
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, environ={})

    assert config.tools.ffmpeg == tmp_path / "tools" / "ffmpeg.exe"
    assert config.asr.model == tmp_path / "models" / "model.bin"
    assert config.output.workspace == tmp_path / "generated"
    assert config.source_path == config_path


def test_environment_overrides_yaml(tmp_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text("asr:\n  device: cpu\n", encoding="utf-8")

    config = load_config(
        config_path,
        environ={
            "VIDEO_MCP_ASR_DEVICE": "cuda",
            "VIDEO_MCP_MAX_LINES": "3",
        },
    )

    assert config.asr.device == "cuda"
    assert config.subtitles.max_lines == 3


@pytest.mark.parametrize("device", ["gpu", "automatic", ""])
def test_invalid_device_is_rejected(tmp_path, device):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text(f"asr:\n  device: {device!r}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="asr.device"):
        load_config(config_path, environ={})
