"""Typed errors raised by the local video processing pipeline."""

from __future__ import annotations

from collections.abc import Sequence


class VideoMcpError(Exception):
    """Base class for expected application errors."""


class ExecutableNotFound(VideoMcpError, FileNotFoundError):
    """Raised when a configured external executable cannot be started."""


class InputFileNotFound(VideoMcpError, FileNotFoundError):
    """Raised when an input media file does not exist."""


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
