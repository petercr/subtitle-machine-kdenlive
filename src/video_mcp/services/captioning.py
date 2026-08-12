"""End-to-end local caption pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr.factory import create_asr_backend
from video_mcp.config import AppConfig
from video_mcp.media.ffmpeg import extract_audio
from video_mcp.media.probe import probe_video
from video_mcp.media.render import create_preview
from video_mcp.models import MediaInfo, Transcript
from video_mcp.logging_config import get_job_logger, new_job_id
from video_mcp.services.cleanup import clean_transcript
from video_mcp.subtitles.ass import write_ass
from video_mcp.subtitles.srt import write_srt

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class CaptionOptions:
    """Runtime choices for one captioning job."""

    language: str = "auto"
    device: str = "auto"
    threads: int = 4
    style: str = "clean"
    preview_width: int = 1280
    create_preview: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.threads <= 0:
            raise ValueError("threads must be greater than zero")
        if self.preview_width <= 0:
            raise ValueError("preview_width must be greater than zero")


@dataclass(frozen=True, slots=True)
class CaptionResult:
    """Paths and metadata produced by a complete captioning job."""

    input_path: Path
    job_id: str
    job_dir: Path
    source_json: Path
    audio_wav: Path
    transcript_raw_json: Path
    transcript_cleaned_json: Path
    subtitles_srt: Path
    subtitles_ass: Path
    preview_mp4: Path | None
    language: str | None
    segment_count: int
    warnings: list[str]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable job result."""

        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, Path):
                value[key] = str(item)
        return value


def caption_video(
    input_path: PathLike,
    config: AppConfig,
    options: CaptionOptions | None = None,
) -> CaptionResult:
    """Run inspect → extract → transcribe → export → preview."""

    options = options or CaptionOptions(style=config.subtitles.preset)
    source = Path(input_path).expanduser().resolve()
    job_id = new_job_id()
    logger = get_job_logger(
        "video_mcp.captioning",
        job_id=job_id,
        input_path=str(source),
        backend=config.asr.backend,
        device=options.device,
    )
    logger.info("Caption job started")
    media = probe_video(source, ffprobe_path=config.tools.ffprobe)
    logger.info("Media inspection completed", extra={"duration_ms": media.duration_ms})
    job_dir = config.output.workspace / source.stem
    job_dir.mkdir(parents=True, exist_ok=True)

    source_json = job_dir / "source.json"
    audio_wav = job_dir / "audio.wav"
    transcript_raw_json = job_dir / "transcript.raw.json"
    transcript_cleaned_json = job_dir / "transcript.cleaned.json"
    subtitles_srt = job_dir / "subtitles.srt"
    subtitles_ass = job_dir / "subtitles.ass"
    preview_mp4 = job_dir / "captioned-preview.mp4"

    _write_json_if_needed(
        source_json,
        {"schema_version": 1, "input": str(source), "media": media.as_dict()},
        overwrite=options.overwrite,
    )
    logger.info("Source metadata ready", extra={"path": str(source_json)})
    if options.overwrite or not audio_wav.exists():
        extract_audio(
            source,
            audio_wav,
            ffmpeg_path=config.tools.ffmpeg,
            overwrite=options.overwrite,
        )
        logger.info("Audio extraction completed", extra={"path": str(audio_wav)})
    else:
        logger.info("Audio extraction reused", extra={"path": str(audio_wav)})

    if options.overwrite or not transcript_raw_json.exists():
        backend = create_asr_backend(config)
        transcript = backend.transcribe(
            audio_wav,
            TranscriptionOptions(
                language=options.language,
                device=options.device,
                threads=options.threads,
            ),
        )
        _write_json(transcript_raw_json, transcript.as_dict())
        logger.info(
            "Transcription completed",
            extra={"path": str(transcript_raw_json), "segment_count": len(transcript.segments)},
        )
    else:
        transcript = _read_transcript(transcript_raw_json)
        logger.info(
            "Transcription reused",
            extra={"path": str(transcript_raw_json), "segment_count": len(transcript.segments)},
        )

    cleanup_warnings: list[str] = []
    if options.overwrite or not transcript_cleaned_json.exists():
        cleanup_result = clean_transcript(transcript, config)
        cleaned_transcript = cleanup_result.transcript
        cleanup_warnings = cleanup_result.warnings
        _write_json(transcript_cleaned_json, cleaned_transcript.as_dict())
        logger.info(
            "Transcript cleanup completed",
            extra={"path": str(transcript_cleaned_json), "warnings": cleanup_warnings},
        )
    else:
        cleaned_transcript = _read_transcript(transcript_cleaned_json)
        logger.info("Transcript cleanup reused", extra={"path": str(transcript_cleaned_json)})
    if options.overwrite or not subtitles_srt.exists():
        write_srt(cleaned_transcript, subtitles_srt, overwrite=options.overwrite)
        logger.info("SRT export completed", extra={"path": str(subtitles_srt)})
    else:
        logger.info("SRT export reused", extra={"path": str(subtitles_srt)})
    if options.overwrite or not subtitles_ass.exists():
        write_ass(
            cleaned_transcript,
            subtitles_ass,
            style=options.style,
            play_res_x=media.video.width if media.video and media.video.width else 1920,
            play_res_y=media.video.height if media.video and media.video.height else 1080,
            overwrite=options.overwrite,
        )
        logger.info("ASS export completed", extra={"path": str(subtitles_ass)})
    else:
        logger.info("ASS export reused", extra={"path": str(subtitles_ass)})
    if options.create_preview:
        if options.overwrite or not preview_mp4.exists():
            create_preview(
                source,
                subtitles_ass,
                preview_mp4,
                ffmpeg_path=config.tools.ffmpeg,
                preview_width=options.preview_width,
                overwrite=options.overwrite,
            )
            logger.info("Preview rendering completed", extra={"path": str(preview_mp4)})
        else:
            logger.info("Preview rendering reused", extra={"path": str(preview_mp4)})
        preview_result: Path | None = preview_mp4
    else:
        preview_result = None

    result = CaptionResult(
        input_path=source,
        job_id=job_id,
        job_dir=job_dir.resolve(),
        source_json=source_json.resolve(),
        audio_wav=audio_wav.resolve(),
        transcript_raw_json=transcript_raw_json.resolve(),
        transcript_cleaned_json=transcript_cleaned_json.resolve(),
        subtitles_srt=subtitles_srt.resolve(),
        subtitles_ass=subtitles_ass.resolve(),
        preview_mp4=preview_result.resolve() if preview_result else None,
        language=cleaned_transcript.language,
        segment_count=len(cleaned_transcript.segments),
        warnings=cleanup_warnings,
    )
    logger.info(
        "Caption job completed",
        extra={"job_dir": str(result.job_dir), "segment_count": result.segment_count},
    )
    return result


def _read_transcript(path: Path) -> Transcript:
    return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_json_if_needed(
    path: Path, value: dict[str, object], *, overwrite: bool
) -> None:
    if overwrite or not path.exists():
        _write_json(path, value)
