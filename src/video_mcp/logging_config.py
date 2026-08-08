"""Structured JSON logging with per-job context."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, TextIO
from uuid import uuid4

_STANDARD_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    """Format one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_FIELDS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str | int = "INFO", *, stream: TextIO | None = None
) -> None:
    """Configure the application root logger for structured output."""

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("video_mcp")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def new_job_id() -> str:
    """Return a compact, collision-resistant processing job identifier."""

    return uuid4().hex


def get_job_logger(
    name: str, *, job_id: str | None = None, **context: Any
) -> logging.LoggerAdapter[logging.Logger]:
    """Return a logger carrying a job ID and optional structured context."""

    extra = {"job_id": job_id or new_job_id(), **context}
    return logging.LoggerAdapter(logging.getLogger(name), extra)
