import json

import pytest
from mcp.types import ImageContent, TextContent

from voice_analysis_mcp.server import mcp

EXPECTED_TOOLS = {
    "get_audio_info",
    "transcribe",
    "analyze_conversation",
    "analyze_prosody",
    "analyze_quality",
    "extract_segment",
    "render_spectrogram",
    "render_waveform",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_tools_registered():
    tools = await mcp.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t.description, f"tool {t.name} is missing a description"


@pytest.mark.anyio
async def test_call_json_tool(stereo_call):
    content = await mcp.call_tool("get_audio_info", {"path": stereo_call})
    assert isinstance(content[0], TextContent)
    assert json.loads(content[0].text)["channels"] == 2


@pytest.mark.anyio
async def test_call_image_tool(stereo_call):
    content = await mcp.call_tool("render_waveform", {"path": stereo_call, "end_time": 3})
    assert isinstance(content[0], ImageContent)
    assert content[0].mimeType == "image/png"


@pytest.mark.anyio
async def test_tool_errors_are_reported(tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="File not found"):
        await mcp.call_tool("get_audio_info", {"path": str(tmp_path / "missing.wav")})


@pytest.mark.anyio
async def test_extract_segment_tool_writes_file(tone_with_silence, tmp_path):
    out = tmp_path / "cut.wav"
    content = await mcp.call_tool(
        "extract_segment",
        {"path": tone_with_silence, "start_time": 1, "end_time": 2, "output_path": str(out)},
    )
    assert json.loads(content[0].text)["output_path"] == str(out)
    assert out.exists()
