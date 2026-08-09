"""Automatic speech recognition backends."""

from video_mcp.asr.base import ASRBackend, TranscriptionOptions
from video_mcp.asr.factory import create_asr_backend
from video_mcp.asr.parakeet import ParakeetBackend, parse_parakeet_output
from video_mcp.asr.whisper_cpp import WhisperCppBackend, parse_whisper_output

__all__ = [
    "ASRBackend",
    "TranscriptionOptions",
    "ParakeetBackend",
    "WhisperCppBackend",
    "create_asr_backend",
    "parse_parakeet_output",
    "parse_whisper_output",
]
