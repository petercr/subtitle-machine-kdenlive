"""Small, dependency-free ASR comparison helpers."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr.factory import create_asr_backend
from video_mcp.config import AppConfig
from video_mcp.models import Transcript

PathLike = str | Path
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ASRBenchmarkResult:
    """Measured output from one ASR backend run."""

    backend: str
    device: str
    input_path: Path
    audio_duration_seconds: float | None
    elapsed_seconds: float
    real_time_factor: float | None
    segment_count: int | None
    word_count: int | None
    language: str | None
    word_error_rate: float | None
    peak_process_memory_mib: float | None
    peak_gpu_memory_mib: float | None
    gpu_memory_scope: str | None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable benchmark record."""

        value = asdict(self)
        value["input_path"] = str(self.input_path)
        return value


class ResourceMonitor:
    """Sample memory consumed by ASR child processes during a benchmark.

    The monitor intentionally uses OS facilities and ``nvidia-smi`` rather
    than adding a Python runtime dependency. Values are best-effort: a missing
    GPU tool or an executable that exits before its first sample simply leaves
    the corresponding metric empty.
    """

    def __init__(self, *, track_gpu: bool, interval_seconds: float = 0.1) -> None:
        self._track_gpu = track_gpu
        self._interval_seconds = interval_seconds
        self._pids: set[int] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_process_memory_mib: float | None = None
        self._peak_gpu_memory_mib: float | None = None
        self._gpu_memory_scope: str | None = None
        self._gpu_baseline_mib: float | None = None

    def start(self) -> None:
        if self._track_gpu:
            self._gpu_baseline_mib = _gpu_device_memory_mib()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def observe_process(self, pid: int) -> None:
        """Register a newly launched ASR process for sampling."""

        with self._lock:
            self._pids.add(pid)
        self._sample_process_memory((pid,))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    @property
    def peak_process_memory_mib(self) -> float | None:
        return self._peak_process_memory_mib

    @property
    def peak_gpu_memory_mib(self) -> float | None:
        return self._peak_gpu_memory_mib

    @property
    def gpu_memory_scope(self) -> str | None:
        return self._gpu_memory_scope

    def _sample_loop(self) -> None:
        next_gpu_sample = 0.0
        while not self._stop_event.wait(self._interval_seconds):
            with self._lock:
                pids = tuple(self._pids)
            if not pids:
                continue
            self._sample_process_memory(pids)
            if self._track_gpu and time.monotonic() >= next_gpu_sample:
                gpu_memory_mib = _gpu_memory_mib(pids)
                if gpu_memory_mib is not None:
                    self._record_gpu_memory(gpu_memory_mib, "process")
                elif self._gpu_baseline_mib is not None:
                    device_memory_mib = _gpu_device_memory_mib()
                    if device_memory_mib is not None:
                        self._record_gpu_memory(
                            max(0, device_memory_mib - self._gpu_baseline_mib),
                            "device_delta",
                        )
                next_gpu_sample = time.monotonic() + 0.5

    def _sample_process_memory(self, pids: tuple[int, ...]) -> None:
        for pid in pids:
            memory_mib = _process_memory_mib(pid)
            if memory_mib is not None:
                self._peak_process_memory_mib = max(self._peak_process_memory_mib or 0, memory_mib)

    def _record_gpu_memory(self, memory_mib: float, scope: str) -> None:
        if self._gpu_memory_scope == "process" and scope != "process":
            return
        self._peak_gpu_memory_mib = max(self._peak_gpu_memory_mib or 0, memory_mib)
        self._gpu_memory_scope = scope


def _process_memory_mib(pid: int) -> float | None:
    """Return resident process memory using Windows or procfs APIs."""

    if os.name == "nt":
        return _windows_process_memory_mib(pid)
    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:") or line.startswith("VmRSS:"):
                kib = int(line.split()[1])
                return kib / 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _windows_process_memory_mib(pid: int) -> float | None:
    """Read Windows working-set bytes without requiring psutil."""

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not process:
        return None
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            return None
        return counters.PeakWorkingSetSize / (1024 * 1024)
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def _gpu_memory_mib(pids: tuple[int, ...]) -> float | None:
    """Return GPU memory used by the watched PIDs according to nvidia-smi."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None

    watched = set(pids)
    total_mib = 0.0
    found = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, memory_mib = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid in watched:
            total_mib += memory_mib
            found = True
    return total_mib if found else None


def _gpu_device_memory_mib() -> float | None:
    """Return aggregate device memory for a labeled WDDM fallback metric."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        values = [float(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    except ValueError:
        return None
    return sum(values) if values else None


def benchmark_asr(
    input_path: PathLike,
    config: AppConfig,
    *,
    reference_text: str | None = None,
    language: str = "auto",
    device: str | None = None,
    threads: int = 4,
) -> ASRBenchmarkResult:
    """Run one configured backend and measure wall-clock performance."""

    audio = Path(input_path).expanduser().resolve()
    duration = audio_duration_seconds(audio)
    backend_name = config.asr.backend
    requested_device = device or config.asr.device
    monitor = ResourceMonitor(track_gpu=requested_device != "cpu")
    monitor.start()
    started = time.perf_counter()
    try:
        transcript = create_asr_backend(config).transcribe(
            audio,
            TranscriptionOptions(
                language=language,
                device=device or config.asr.device,
                threads=threads,
                process_started=monitor.observe_process,
            ),
        )
    except Exception as exc:  # benchmark records failures for comparison
        elapsed = time.perf_counter() - started
        monitor.stop()
        return ASRBenchmarkResult(
            backend=backend_name,
            device=requested_device,
            input_path=audio,
            audio_duration_seconds=duration,
            elapsed_seconds=elapsed,
            real_time_factor=_real_time_factor(elapsed, duration),
            segment_count=None,
            word_count=None,
            language=None,
            word_error_rate=None,
            peak_process_memory_mib=monitor.peak_process_memory_mib,
            peak_gpu_memory_mib=monitor.peak_gpu_memory_mib,
            gpu_memory_scope=monitor.gpu_memory_scope,
            error=str(exc),
        )

    elapsed = time.perf_counter() - started
    monitor.stop()
    hypothesis = transcript_text(transcript)
    return ASRBenchmarkResult(
        backend=backend_name,
        device=requested_device,
        input_path=audio,
        audio_duration_seconds=duration,
        elapsed_seconds=elapsed,
        real_time_factor=_real_time_factor(elapsed, duration),
        segment_count=len(transcript.segments),
        word_count=len(normalize_words(hypothesis)),
        language=transcript.language,
        word_error_rate=(
            word_error_rate(reference_text, hypothesis) if reference_text is not None else None
        ),
        peak_process_memory_mib=monitor.peak_process_memory_mib,
        peak_gpu_memory_mib=monitor.peak_gpu_memory_mib,
        gpu_memory_scope=monitor.gpu_memory_scope,
    )


def audio_duration_seconds(path: PathLike) -> float | None:
    """Read duration for a normalized PCM WAV without external dependencies."""

    try:
        with wave.open(str(Path(path)), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                return None
            return audio.getnframes() / frame_rate
    except (OSError, wave.Error):
        return None


def normalize_words(text: str) -> list[str]:
    """Normalize text for a case- and punctuation-insensitive WER comparison."""

    return [word.lower() for word in _WORD_RE.findall(text)]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate word error rate using Levenshtein edit distance."""

    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous = list(range(len(hypothesis_words) + 1))
    for reference_word in reference_words:
        current = [previous[0] + 1]
        for index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous[index - 1] + (reference_word != hypothesis_word)
            insertion = current[index - 1] + 1
            deletion = previous[index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference_words)


def _real_time_factor(elapsed: float, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    return elapsed / duration


def transcript_text(transcript: Transcript) -> str:
    """Return the plain text used for benchmark comparisons."""

    return " ".join(segment.text for segment in transcript.segments)
