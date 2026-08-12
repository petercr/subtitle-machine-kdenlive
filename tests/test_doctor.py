import json
from pathlib import Path

from video_mcp.cli import main
from video_mcp.config import AppConfig, ASRConfig, OutputConfig, ToolConfig, load_config
from video_mcp.doctor import READY, _whisper_gpu_diagnostic, run_doctor


def _missing_tools_config(workspace: Path) -> AppConfig:
    missing = Path("missing-tool")
    return AppConfig(
        tools=ToolConfig(
            ffmpeg=missing,
            ffprobe=missing,
            whisper_cpp=missing,
            kdenlive=missing,
            melt=missing,
            llama_cpp=missing,
        ),
        asr=ASRConfig(model=workspace / "missing-model.bin"),
        output=OutputConfig(workspace=workspace),
    )


def test_doctor_reports_missing_tools_without_crashing(tmp_path):
    report = run_doctor(_missing_tools_config(tmp_path / "work"))

    statuses = {diagnostic.name: diagnostic.status for diagnostic in report.diagnostics}
    assert report.core_ready is False
    assert statuses["FFmpeg"] == "NOT FOUND"
    assert statuses["Whisper model"] == "NOT FOUND"
    assert statuses["Whisper GPU"] == "NOT INSTALLED"
    assert statuses["llama.cpp"] == "NOT INSTALLED"
    assert statuses["Workspace"] == READY
    assert (tmp_path / "work").is_dir()


def test_doctor_json_output_is_structured(capsys, tmp_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text(
        f"output:\n  workspace: {tmp_path / 'work'}\n", encoding="utf-8"
    )

    assert main(["--config", str(config_path), "doctor", "--json"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["core_caption_pipeline"] == "NOT READY"
    assert "capabilities" in report


def test_doctor_reports_enabled_llm_model_as_optional(tmp_path):
    config_path = tmp_path / "video-mcp.yaml"
    config_path.write_text(
        f"llm:\n  enabled: true\n  model: {tmp_path / 'missing.gguf'}\n",
        encoding="utf-8",
    )

    report = run_doctor(load_config(config_path, environ={}))

    llm_model = next(item for item in report.diagnostics if item.name == "LLM cleanup model")
    assert llm_model.status == "NOT INSTALLED"
    assert llm_model.optional is True


def test_doctor_detects_whisper_cuda_backend_library(tmp_path):
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    executable = whisper_dir / "whisper-cli.exe"
    executable.write_bytes(b"executable")
    (whisper_dir / "ggml-cuda.dll").write_bytes(b"cuda")

    gpu = _whisper_gpu_diagnostic(executable)
    assert gpu.status == READY
    assert "ggml-cuda.dll" in gpu.detail
