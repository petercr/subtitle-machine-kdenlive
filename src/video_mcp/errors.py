"""Typed errors raised by the local video processing pipeline."""

from __future__ import annotations

from collections.abc import Sequence


class VideoMcpError(Exception):
    """Base class for expected application errors."""


class ExecutableNotFound(VideoMcpError, FileNotFoundError):
    """Raised when a configured external executable cannot be started."""


class InputFileNotFound(VideoMcpError, FileNotFoundError):
    """Raised when an input media file does not exist."""


class ModelNotFound(VideoMcpError, FileNotFoundError):
    """Raised when the configured ASR model cannot be found."""


class UnsupportedMedia(VideoMcpError):
    """Raised when FFprobe cannot find a usable video stream."""


class ExternalCommandFailed(VideoMcpError):
    """Base error for an external media command that returned a failure."""

    def __init__(
        self,
        operation: str,
        command: Sequence[str],
        returncode: int,
        stderr: str = "",
    ) -> None:
        self.operation = operation
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr.strip()
        detail = self.stderr or "no diagnostic output"
        super().__init__(
            f"{operation} failed with exit code {returncode}: {detail}"
        )


class MediaProbeFailed(ExternalCommandFailed):
    """Raised when FFprobe cannot inspect the input media."""


class AudioExtractionFailed(ExternalCommandFailed):
    """Raised when FFmpeg cannot create the normalized WAV file."""


class InvalidMediaProbeOutput(VideoMcpError):
    """Raised when FFprobe returns output that is not valid JSON media data."""


class TranscriptionFailed(ExternalCommandFailed):
    """Raised when an ASR backend cannot produce a transcript."""


class InvalidTranscriptOutput(VideoMcpError):
    """Raised when an ASR backend returns malformed structured output."""


class SubtitleGenerationFailed(VideoMcpError):
    """Raised when normalized subtitle data cannot produce a valid file."""


class CleanupFailed(ExternalCommandFailed):
    """Raised when the optional local transcript cleanup command fails."""


class InvalidCleanupOutput(VideoMcpError):
    """Raised when a cleanup model violates the required response schema."""


class RenderFailed(ExternalCommandFailed):
    """Raised when FFmpeg cannot render a subtitled video."""


class KdenliveProjectFailed(VideoMcpError):
    """Raised when a Kdenlive project cannot be generated safely."""
