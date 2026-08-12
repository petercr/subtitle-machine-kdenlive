import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from video_mcp.mcp.server import mcp


EXPECTED_TOOLS = {
    "video.inspect",
    "video.transcribe",
    "video.caption",
    "video.create_preview",
    "video.render",
    "subtitle.clean",
    "subtitle.export_srt",
    "subtitle.export_ass",
    "project.create_kdenlive",
}


def test_mcp_registers_the_initial_tool_surface():
    tools = asyncio.run(mcp.list_tools())

    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    inspect_tool = next(tool for tool in tools if tool.name == "video.inspect")
    assert inspect_tool.input_schema["required"] == ["input_path"]


def test_mcp_dispatches_subtitle_export_with_structured_output(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "language": "en",
                "duration_ms": 1000,
                "segments": [
                    {
                        "id": "seg-1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "Hello MCP",
                        "words": [],
                        "speaker": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "captions.srt"

    result = asyncio.run(
        mcp.call_tool(
            "subtitle.export_srt",
            {
                "transcript_path": str(transcript_path),
                "output_path": str(output_path),
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content["srt_path"] == str(output_path.resolve())
    assert "Hello MCP" in output_path.read_text(encoding="utf-8")


def test_mcp_dispatches_subtitle_clean_with_deterministic_fallback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "language": "en",
                "duration_ms": 1000,
                "segments": [
                    {
                        "id": "seg-1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": " hello ,   mcp",
                        "words": [],
                        "speaker": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "cleaned.json"

    result = asyncio.run(
        mcp.call_tool(
            "subtitle.clean",
            {
                "transcript_path": str(transcript_path),
                "output_path": str(output_path),
            },
        )
    )

    assert result.is_error is False
    assert result.structured_content["used_llm"] is False
    cleaned = json.loads(output_path.read_text(encoding="utf-8"))
    assert cleaned["segments"][0]["text"] == "Hello, mcp"


def test_mcp_stdio_server_initializes_and_lists_tools():
    async def round_trip():
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "video_mcp.mcp.server"],
            cwd=Path.cwd(),
        )
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return {tool.name for tool in result.tools}

    assert asyncio.run(round_trip()) == EXPECTED_TOOLS
