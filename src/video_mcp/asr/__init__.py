"""Local automatic speech recognition backends."""
"""Automatic speech recognition backends."""

from video_mcp.asr.base import ASRBackend, TranscriptionOptions
from video_mcp.asr.whisper_cpp import WhisperCppBackend, parse_whisper_output

__all__ = [
    "ASRBackend",
    "TranscriptionOptions",
    "WhisperCppBackend",
    "parse_whisper_output",
]
