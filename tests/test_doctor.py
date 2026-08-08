import json
from pathlib import Path

from video_mcp.cli import main
from video_mcp.config import AppConfig, ASRConfig, OutputConfig, ToolConfig
from video_mcp.doctor import READY, run_doctor


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
