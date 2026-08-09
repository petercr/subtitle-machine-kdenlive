"""ASR backend selection from application configuration."""

from __future__ import annotations

from video_mcp.asr.base import ASRBackend
from video_mcp.asr.parakeet import ParakeetBackend
from video_mcp.asr.whisper_cpp import WhisperCppBackend
from video_mcp.config import AppConfig


def create_asr_backend(config: AppConfig) -> ASRBackend:
    """Create the configured local ASR backend."""

    backend = config.asr.backend.lower()
    if backend == "whisper_cpp":
        return WhisperCppBackend(config.tools.whisper_cpp, config.asr.model)
    if backend == "parakeet":
        return ParakeetBackend(config.tools.parakeet, config.asr.model)
    raise ValueError(f"Unsupported ASR backend: {config.asr.backend}")
