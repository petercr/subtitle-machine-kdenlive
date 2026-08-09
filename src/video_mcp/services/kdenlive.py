"""Application service for creating editable Kdenlive projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from video_mcp.adapters.kdenlive import KdenliveProjectAdapter
from video_mcp.config import AppConfig
from video_mcp.errors import KdenliveProjectFailed
from video_mcp.media.probe import probe_video

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class KdenliveProjectResult:
    """Paths and source metadata produced by an editable project export."""

    input_path: Path
    subtitles_path: Path
    project_path: Path
    project_subtitles_path: Path
    duration_ms: int
    width: int
    height: int
    frame_rate: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable project result."""

        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, Path):
                value[key] = str(item)
        return value


def create_kdenlive_project(
    input_path: PathLike,
    subtitles_path: PathLike,
    config: AppConfig,
    *,
    output_path: PathLike | None = None,
    overwrite: bool = False,
) -> KdenliveProjectResult:
    """Probe a video and write a Kdenlive project using an SRT subtitle file."""

    source = Path(input_path).expanduser().resolve()
    subtitles = Path(subtitles_path).expanduser().resolve()
    if subtitles.suffix.lower() != ".srt":
        raise KdenliveProjectFailed(
            "Editable Kdenlive subtitle export requires an .srt file"
        )

    media = probe_video(source, ffprobe_path=config.tools.ffprobe)
    if media.video is None or media.video.width is None or media.video.height is None:
        raise KdenliveProjectFailed("Kdenlive export requires video dimensions")
    if media.video.frame_rate is None:
        raise KdenliveProjectFailed("Kdenlive export requires a video frame rate")
    if media.duration_ms is None:
        raise KdenliveProjectFailed("Kdenlive export requires a media duration")

    destination = Path(output_path).expanduser() if output_path else (
        config.output.workspace / f"{source.stem}-captioned.kdenlive"
    )
    destination = destination.resolve()

    adapter = KdenliveProjectAdapter()
    project = adapter.create_project(source, media)
    adapter.add_subtitles(project, subtitles)
    project_path = adapter.save(project, destination, overwrite=overwrite)
    project_subtitles_path = Path(f"{project_path}.srt")

    return KdenliveProjectResult(
        input_path=source,
        subtitles_path=subtitles,
        project_path=project_path,
        project_subtitles_path=project_subtitles_path,
        duration_ms=media.duration_ms,
        width=media.video.width,
        height=media.video.height,
        frame_rate=media.video.frame_rate,
    )
