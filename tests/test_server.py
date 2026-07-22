import pytest

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


@pytest.mark.anyio
async def test_tools_registered():
    tools = await mcp.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t.description, f"tool {t.name} is missing a description"


@pytest.fixture
def anyio_backend():
    return "asyncio"
