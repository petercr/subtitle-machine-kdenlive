"""End-to-end local caption pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from video_mcp.asr.base import TranscriptionOptions
from video_mcp.asr.whisper_cpp import WhisperCppBackend
from video_mcp.config import AppConfig
from video_mcp.media.ffmpeg import extract_audio
from video_mcp.media.probe import probe_video
from video_mcp.media.render import create_preview
from video_mcp.models import MediaInfo, Transcript
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
    media = probe_video(source, ffprobe_path=config.tools.ffprobe)
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
    if options.overwrite or not audio_wav.exists():
        extract_audio(
            source,
            audio_wav,
            ffmpeg_path=config.tools.ffmpeg,
            overwrite=options.overwrite,
        )

    if options.overwrite or not transcript_raw_json.exists():
        backend = WhisperCppBackend(config.tools.whisper_cpp, config.asr.model)
        transcript = backend.transcribe(
            audio_wav,
            TranscriptionOptions(
                language=options.language,
                device=options.device,
                threads=options.threads,
            ),
        )
        _write_json(transcript_raw_json, transcript.as_dict())
    else:
        transcript = _read_transcript(transcript_raw_json)

    if options.overwrite or not transcript_cleaned_json.exists():
        _write_json(transcript_cleaned_json, transcript.as_dict())
    if options.overwrite or not subtitles_srt.exists():
        write_srt(transcript, subtitles_srt, overwrite=options.overwrite)
    if options.overwrite or not subtitles_ass.exists():
        write_ass(
            transcript,
            subtitles_ass,
            style=options.style,
            play_res_x=media.video.width if media.video and media.video.width else 1920,
            play_res_y=media.video.height if media.video and media.video.height else 1080,
            overwrite=options.overwrite,
        )
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
        preview_result: Path | None = preview_mp4
    else:
        preview_result = None

    return CaptionResult(
        input_path=source,
        job_dir=job_dir.resolve(),
        source_json=source_json.resolve(),
        audio_wav=audio_wav.resolve(),
        transcript_raw_json=transcript_raw_json.resolve(),
        transcript_cleaned_json=transcript_cleaned_json.resolve(),
        subtitles_srt=subtitles_srt.resolve(),
        subtitles_ass=subtitles_ass.resolve(),
        preview_mp4=preview_result.resolve() if preview_result else None,
        language=transcript.language,
        segment_count=len(transcript.segments),
        warnings=["Deterministic cleanup is not configured; cleaned transcript matches raw transcript."],
    )


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
