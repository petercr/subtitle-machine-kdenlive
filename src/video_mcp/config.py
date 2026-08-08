"""Application configuration loaded from YAML and environment variables."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_FILENAME = "video-mcp.yaml"


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ToolConfig:
    ffmpeg: Path = Path("ffmpeg")
    ffprobe: Path = Path("ffprobe")
    whisper_cpp: Path = Path("whisper-cli")
    kdenlive: Path = Path("kdenlive")
    melt: Path = Path("melt")
    llama_cpp: Path = Path("llama-cli")


@dataclass(frozen=True, slots=True)
class ASRConfig:
    backend: str = "whisper_cpp"
    device: str = "auto"
    model: Path = Path("C:/Models/whisper/ggml-small.bin")


@dataclass(frozen=True, slots=True)
class SubtitleConfig:
    preset: str = "clean"
    max_chars_per_line: int = 42
    max_lines: int = 2


@dataclass(frozen=True, slots=True)
class OutputConfig:
    workspace: Path = Path("work")


@dataclass(frozen=True, slots=True)
class AppConfig:
    tools: ToolConfig = ToolConfig()
    asr: ASRConfig = ASRConfig()
    subtitles: SubtitleConfig = SubtitleConfig()
    output: OutputConfig = OutputConfig()
    source_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return _serialize_paths(asdict(self))


ENVIRONMENT_OVERRIDES: dict[str, tuple[str, str]] = {
    "VIDEO_MCP_FFMPEG": ("tools", "ffmpeg"),
    "VIDEO_MCP_FFPROBE": ("tools", "ffprobe"),
    "VIDEO_MCP_WHISPER_CPP": ("tools", "whisper_cpp"),
    "VIDEO_MCP_KDENLIVE": ("tools", "kdenlive"),
    "VIDEO_MCP_MELT": ("tools", "melt"),
    "VIDEO_MCP_LLAMA_CPP": ("tools", "llama_cpp"),
    "VIDEO_MCP_ASR_BACKEND": ("asr", "backend"),
    "VIDEO_MCP_ASR_DEVICE": ("asr", "device"),
    "VIDEO_MCP_ASR_MODEL": ("asr", "model"),
    "VIDEO_MCP_SUBTITLE_PRESET": ("subtitles", "preset"),
    "VIDEO_MCP_MAX_CHARS_PER_LINE": ("subtitles", "max_chars_per_line"),
    "VIDEO_MCP_MAX_LINES": ("subtitles", "max_lines"),
    "VIDEO_MCP_WORKSPACE": ("output", "workspace"),
}


def load_config(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load configuration, with environment variables taking precedence."""

    explicit_path = config_path is not None
    path = Path(config_path) if explicit_path else Path.cwd() / DEFAULT_CONFIG_FILENAME
    path = path.expanduser()

    raw: dict[str, Any] = {}
    source_path: Path | None = None
    if path.exists():
        source_path = path.resolve()
        raw = _read_yaml(source_path)
    elif explicit_path:
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    env = os.environ if environ is None else environ
    merged = _apply_environment_overrides(raw, env)
    base_dir = source_path.parent if source_path is not None else Path.cwd()
    return _build_config(merged, base_dir=base_dir, source_path=source_path)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read configuration {path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping.")
    return loaded


def _apply_environment_overrides(
    raw: Mapping[str, Any], environ: Mapping[str, str]
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in raw.items()
    }
    for env_name, (section_name, key) in ENVIRONMENT_OVERRIDES.items():
        if env_name not in environ:
            continue
        section = merged.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ConfigurationError(f"Configuration section '{section_name}' must be a mapping.")
        section[key] = environ[env_name]
    return merged


def _build_config(
    raw: Mapping[str, Any], *, base_dir: Path, source_path: Path | None
) -> AppConfig:
    tools = _section(raw, "tools")
    asr = _section(raw, "asr")
    subtitles = _section(raw, "subtitles")
    output = _section(raw, "output")

    device = str(asr.get("device", "auto")).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("asr.device must be one of: auto, cpu, cuda")

    max_chars = _positive_int(
        subtitles.get("max_chars_per_line", 42), "subtitles.max_chars_per_line"
    )
    max_lines = _positive_int(subtitles.get("max_lines", 2), "subtitles.max_lines")

    return AppConfig(
        tools=ToolConfig(
            ffmpeg=_tool_path(tools.get("ffmpeg", "ffmpeg"), base_dir),
            ffprobe=_tool_path(tools.get("ffprobe", "ffprobe"), base_dir),
            whisper_cpp=_tool_path(tools.get("whisper_cpp", "whisper-cli"), base_dir),
            kdenlive=_tool_path(tools.get("kdenlive", "kdenlive"), base_dir),
            melt=_tool_path(tools.get("melt", "melt"), base_dir),
            llama_cpp=_tool_path(tools.get("llama_cpp", "llama-cli"), base_dir),
        ),
        asr=ASRConfig(
            backend=str(asr.get("backend", "whisper_cpp")),
            device=device,
            model=_resolved_path(
                asr.get("model", "C:/Models/whisper/ggml-small.bin"), base_dir
            ),
        ),
        subtitles=SubtitleConfig(
            preset=str(subtitles.get("preset", "clean")),
            max_chars_per_line=max_chars,
            max_lines=max_lines,
        ),
        output=OutputConfig(
            workspace=_resolved_path(output.get("workspace", "work"), base_dir)
        ),
        source_path=source_path,
    )


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping.")
    return value

def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be an integer.") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{field_name} must be greater than zero.")
    return parsed


def _tool_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() or any(separator in str(value) for separator in ("/", "\\")):
        return _resolve_relative(path, base_dir)
    return path


def _resolved_path(value: Any, base_dir: Path) -> Path:
    return _resolve_relative(Path(str(value)).expanduser(), base_dir)


def _resolve_relative(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _serialize_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_paths(item) for item in value]
    return value
