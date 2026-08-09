from pathlib import Path
from xml.etree import ElementTree

import pytest

from video_mcp.adapters.kdenlive import KdenliveProjectAdapter
from video_mcp.errors import KdenliveProjectFailed
from video_mcp.models import AudioStreamInfo, MediaInfo, VideoStreamInfo


def _media(source: Path, *, frame_rate: float = 29.97) -> MediaInfo:
    return MediaInfo(
        path=source,
        duration_ms=2500,
        format_name="mp4",
        size_bytes=123,
        bit_rate=1000,
        video=VideoStreamInfo("h264", 1920, 1080, frame_rate, None),
        audio=AudioStreamInfo("aac", 48000, 2, "stereo"),
    )


def _properties(element: ElementTree.Element) -> dict[str, str]:
    return {
        child.attrib["name"]: child.text or ""
        for child in element.findall("property")
    }


def test_kdenlive_writer_creates_gen5_project_and_sibling_srt(tmp_path):
    source = tmp_path / "Source Videos" / "My & source video.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    subtitles = tmp_path / "captions.srt"
    subtitles.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    destination = tmp_path / "Project Output" / "captioned.kdenlive"

    adapter = KdenliveProjectAdapter()
    project = adapter.create_project(source, _media(source))
    adapter.add_subtitles(project, subtitles)
    result = adapter.save(project, destination)

    sibling = Path(f"{result}.srt")
    assert result == destination.resolve()
    assert result.is_file()
    assert sibling.read_text(encoding="utf-8") == subtitles.read_text(encoding="utf-8")

    root = ElementTree.parse(result).getroot()
    assert root.tag == "mlt"
    assert root.attrib["producer"] == "main_bin"
    assert root.attrib["LC_NUMERIC"] == "C"

    profile = root.find("profile")
    assert profile is not None
    assert profile.attrib["frame_rate_num"] == "30000"
    assert profile.attrib["frame_rate_den"] == "1001"
    assert profile.attrib["width"] == "1920"
    assert profile.attrib["height"] == "1080"

    producer = root.find("producer[@id='producer0']")
    assert producer is not None
    producer_properties = _properties(producer)
    assert producer_properties["resource"] == source.resolve().as_posix()
    assert producer_properties["kdenlive:clipname"] == source.name

    main_bin = root.find("playlist[@id='main_bin']")
    assert main_bin is not None
    assert _properties(main_bin)["kdenlive:docproperties.version"] == "1.1"
    assert [entry.attrib["producer"] for entry in main_bin.findall("entry")] == [
        "producer0",
        "tractor1",
    ]

    sequence = root.find("tractor[@id='tractor1']")
    assert sequence is not None
    subtitle_filter = sequence.find("filter[@id='filter0']")
    assert subtitle_filter is not None
    subtitle_properties = _properties(subtitle_filter)
    assert subtitle_properties["mlt_service"] == "avfilter.subtitles"
    assert subtitle_properties["internal_added"] == "237"
    assert subtitle_properties["kdenlive:locked"] == "1"
    assert subtitle_properties["av.filename"] == sibling.as_posix()
    adapter.save(project, destination, overwrite=True)
    refreshed_root = ElementTree.parse(result).getroot()
    refreshed_filter = refreshed_root.find("tractor[@id='tractor1']/filter[@id='filter0']")
    assert refreshed_filter is not None
    assert len(
        [
            child
            for child in refreshed_filter.findall("property")
            if child.attrib.get("name") == "av.filename"
        ]
    ) == 1

    main_tractor = refreshed_root.findall("tractor")[-1]
    assert main_tractor.attrib["id"] == "maintractor"
    assert _properties(main_tractor)["kdenlive:projectTractor"] == "1"


def test_kdenlive_writer_preserves_existing_project_by_default(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    destination = tmp_path / "captioned.kdenlive"
    adapter = KdenliveProjectAdapter()

    project = adapter.create_project(source, _media(source, frame_rate=30.0))
    adapter.save(project, destination)

    with pytest.raises(KdenliveProjectFailed, match="Output already exists"):
        adapter.save(project, destination)


def test_kdenlive_writer_requires_complete_video_metadata(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    media = MediaInfo(
        path=source,
        duration_ms=1000,
        format_name="mp4",
        size_bytes=None,
        bit_rate=None,
        video=VideoStreamInfo("h264", None, 1080, 30.0, None),
        audio=None,
    )

    with pytest.raises(KdenliveProjectFailed, match="video dimensions"):
        KdenliveProjectAdapter().create_project(source, media)


def test_kdenlive_writer_requires_kdenlive_extension(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    project = KdenliveProjectAdapter().create_project(source, _media(source))

    with pytest.raises(KdenliveProjectFailed, match=r"\.kdenlive extension"):
        KdenliveProjectAdapter().save(project, tmp_path / "project.xml")
