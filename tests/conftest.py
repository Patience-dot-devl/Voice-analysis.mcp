import shutil
import subprocess
from pathlib import Path

import pytest


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


@pytest.fixture(scope="session")
def stereo_call(tmp_path_factory) -> str:
    """20s stereo file: channel 0 active seconds 0-3, 6-9, ...; channel 1 on 3-6, 9-12, ..."""
    out = tmp_path_factory.mktemp("audio") / "stereo_call.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=220:duration=20",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
        "-filter_complex",
        "[0:a]volume='if(lt(mod(t,6),3),1,0)':eval=frame[a];"
        "[1:a]volume='if(lt(mod(t,6),3),0,1)':eval=frame[b];"
        "[a][b]join=inputs=2:channel_layout=stereo",
        "-ar", "16000", str(out),
    )
    return str(out)


@pytest.fixture(scope="session")
def tone_with_silence(tmp_path_factory) -> str:
    """10s mono: 4s tone, 3s silence, 3s tone."""
    out = tmp_path_factory.mktemp("audio") / "tone_gap.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=300:duration=10",
        "-af", "volume='if(between(t,4,7),0,0.5)':eval=frame",
        "-ar", "16000", "-ac", "1", str(out),
    )
    return str(out)


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory) -> str:
    """Synthesized speech via macOS `say` (skipped elsewhere)."""
    if shutil.which("say") is None:
        pytest.skip("macOS `say` not available for speech synthesis")
    d = tmp_path_factory.mktemp("audio")
    aiff = d / "speech.aiff"
    out = d / "speech.wav"
    subprocess.run(
        ["say", "-o", str(aiff), "Hello, thank you for calling customer support. How can I help you today?"],
        check=True,
    )
    _ffmpeg("-i", str(aiff), "-ar", "16000", "-ac", "1", str(out))
    return str(out)
