from pathlib import Path

import pytest

from video_mcp.config import ConfigurationError, load_config


def test_defaults_are_resolved_from_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_config(environ={})

    assert config.tools.ffmpeg == Path("ffmpeg")
    assert config.asr.device == "auto"
    assert config.asr.model == Path("C:/Models/whisper/ggml-small.bin")
    assert config.llm.enabled is False
    assert config.llm.model == Path("C:/Models/llama/Qwen3.5-2B-Q4_K_M.gguf")
    assert config.llm.server_url is None
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


@pytest.mark.parametrize(
    "model_path",
    ["C:/Models/whisper/model.bin", r"\\server\models\whisper.bin"],
)
def test_windows_absolute_paths_are_not_resolved_relative_to_configuration(tmp_path, model_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text(
        f"asr:\n  model: '{model_path}'\n",
        encoding="utf-8",
    )

    config = load_config(config_path, environ={})

    assert str(config.asr.model).replace("\\", "/") == model_path.replace("\\", "/")


def test_environment_overrides_yaml(tmp_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text("asr:\n  device: cpu\n", encoding="utf-8")

    config = load_config(
        config_path,
        environ={
            "VIDEO_MCP_ASR_DEVICE": "cuda",
            "VIDEO_MCP_MAX_LINES": "3",
            "VIDEO_MCP_LLM_ENABLED": "true",
            "VIDEO_MCP_LLM_MAX_SEGMENTS": "4",
            "VIDEO_MCP_LLM_SERVER_URL": "http://127.0.0.1:8087/v1/chat/completions",
        },
    )

    assert config.asr.device == "cuda"
    assert config.subtitles.max_lines == 3
    assert config.llm.enabled is True
    assert config.llm.max_segments_per_chunk == 4
    assert config.llm.server_url == "http://127.0.0.1:8087/v1/chat/completions"


def test_invalid_llm_enabled_is_rejected(tmp_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text("llm:\n  enabled: maybe\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="llm.enabled"):
        load_config(config_path, environ={})


def test_invalid_llm_server_url_is_rejected(tmp_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text("llm:\n  server_url: llama://local\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="llm.server_url"):
        load_config(config_path, environ={})


@pytest.mark.parametrize("device", ["gpu", "automatic", ""])
def test_invalid_device_is_rejected(tmp_path, device):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text(f"asr:\n  device: {device!r}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="asr.device"):
        load_config(config_path, environ={})
