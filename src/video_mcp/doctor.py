"""Environment diagnostics for the local video subtitle pipeline."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from video_mcp.config import AppConfig

READY = "READY"
NOT_FOUND = "NOT FOUND"
NOT_DETECTED = "NOT DETECTED"
NOT_INSTALLED = "NOT INSTALLED"
ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One capability check from the doctor report."""

    name: str
    status: str
    detail: str
    optional: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Complete diagnostic result for the local processing environment."""

    windows: str
    cpu: str
    logical_processors: int
    ram_gb: float | None
    diagnostics: tuple[Diagnostic, ...]

    @property
    def core_ready(self) -> bool:
        required = {"FFmpeg", "FFprobe", "Whisper.cpp", "Whisper model", "Workspace"}
        statuses = {diagnostic.name: diagnostic.status for diagnostic in self.diagnostics}
        return all(statuses.get(name) == READY for name in required)

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": {
                "windows": self.windows,
                "cpu": self.cpu,
                "logical_processors": self.logical_processors,
                "ram_gb": self.ram_gb,
            },
            "core_caption_pipeline": READY if self.core_ready else "NOT READY",
            "capabilities": [diagnostic.as_dict() for diagnostic in self.diagnostics],
        }


def run_doctor(config: AppConfig) -> DoctorReport:
    """Inspect configured tools, optional accelerators, and the workspace."""

    diagnostics = (
        _tool_diagnostic("FFmpeg", config.tools.ffmpeg, args=("-version",)),
        _tool_diagnostic("FFprobe", config.tools.ffprobe, args=("-version",)),
        _tool_diagnostic("Whisper.cpp", config.tools.whisper_cpp, args=("-h",)),
        _model_diagnostic(config.asr.model),
        _nvidia_diagnostic(),
        _tool_diagnostic("Kdenlive", config.tools.kdenlive),
        _tool_diagnostic("MLT/melt", config.tools.melt),
        _tool_diagnostic("Parakeet", config.tools.parakeet, args=("-h",), optional=True),
        _tool_diagnostic("llama.cpp", config.tools.llama_cpp, optional=True),
        *(
            (_model_diagnostic(config.llm.model, name="LLM cleanup model", optional=True),)
            if config.llm.enabled
            else ()
        ),
        _workspace_diagnostic(config.output.workspace),
    )

    return DoctorReport(
        windows=f"{platform.system()} {platform.release()} ({platform.version()})",
        cpu=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine(),
        logical_processors=os.cpu_count() or 0,
        ram_gb=_total_memory_gb(),
        diagnostics=diagnostics,
    )


def format_doctor_report(report: DoctorReport) -> str:
    """Render a concise human-readable doctor report."""

    ram = f"{report.ram_gb:.1f} GB" if report.ram_gb is not None else "unknown"
    lines = [
        "Video Subtitle MCP Doctor",
        "",
        f"Windows: {report.windows}",
        f"CPU: {report.cpu}",
        f"Logical processors: {report.logical_processors}",
        f"System RAM: {ram}",
        "",
        f"Core caption pipeline: {'READY' if report.core_ready else 'NOT READY'}",
    ]
    for diagnostic in report.diagnostics:
        optional = " (optional)" if diagnostic.optional else ""
        lines.append(f"{diagnostic.name}{optional}: {diagnostic.status} — {diagnostic.detail}")
    return "\n".join(lines)


def _tool_diagnostic(
    name: str,
    configured_path: Path,
    *,
    args: Sequence[str] = ("--version",),
    optional: bool = False,
) -> Diagnostic:
    executable = _resolve_executable(configured_path)
    if executable is None:
        status = NOT_INSTALLED if optional else NOT_FOUND
        return Diagnostic(name, status, f"Configured path: {configured_path}", optional)

    try:
        result = subprocess.run(
            [str(executable), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Diagnostic(name, ERROR, f"{executable} ({exc})", optional)

    output = _first_line(result.stdout) or _first_line(result.stderr)
    if result.returncode != 0:
        detail = output or f"exited with code {result.returncode}"
        return Diagnostic(name, ERROR, f"{executable} — {detail}", optional)
    return Diagnostic(name, READY, f"{executable} — {output or 'available'}", optional)


def _model_diagnostic(
    model_path: Path, *, name: str = "Whisper model", optional: bool = False
) -> Diagnostic:
    if not model_path.is_file():
        status = NOT_INSTALLED if optional else NOT_FOUND
        return Diagnostic(name, status, f"Configured path: {model_path}", optional)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    return Diagnostic(name, READY, f"{model_path} ({size_mb:.0f} MiB)", optional)


def _nvidia_diagnostic() -> Diagnostic:
    executable = _resolve_executable(Path("nvidia-smi"))
    if executable is None:
        return Diagnostic("NVIDIA GPU", NOT_DETECTED, "nvidia-smi was not found", optional=True)

    try:
        result = subprocess.run(
            [
                str(executable),
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Diagnostic("NVIDIA GPU", ERROR, str(exc), optional=True)

    output = _first_line(result.stdout) or _first_line(result.stderr)
    if result.returncode != 0:
        return Diagnostic("NVIDIA GPU", ERROR, output or "nvidia-smi failed", optional=True)
    return Diagnostic("NVIDIA GPU", READY, output or "detected", optional=True)


def _workspace_diagnostic(workspace: Path) -> Diagnostic:
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=".doctor-", suffix=".tmp", dir=workspace, delete=False
        ) as probe:
            probe.write("workspace probe\n")
            probe_path = Path(probe.name)
        probe_path.unlink()
    except (OSError, ValueError) as exc:
        return Diagnostic("Workspace", ERROR, f"{workspace} ({exc})")
    return Diagnostic("Workspace", READY, str(workspace))


def _resolve_executable(configured_path: Path) -> Path | None:
    path_text = str(configured_path)
    if configured_path.is_absolute() or any(separator in path_text for separator in ("/", "\\")):
        return configured_path if configured_path.is_file() else None
    resolved = shutil.which(path_text)
    return Path(resolved) if resolved else None


def _first_line(value: str) -> str:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _total_memory_gb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    try:
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except AttributeError:
        return None
    if not success:
        return None
    return round(status.ullTotalPhys / (1024**3), 1)
