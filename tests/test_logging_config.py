import io
import json

from video_mcp.logging_config import configure_logging, get_job_logger


def test_job_logger_emits_structured_json():
    stream = io.StringIO()
    configure_logging(stream=stream)
    logger = get_job_logger("video_mcp.test", job_id="job-123", input="demo.mp4")

    logger.info("Started")

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "INFO"
    assert payload["message"] == "Started"
    assert payload["job_id"] == "job-123"
    assert payload["input"] == "demo.mp4"
