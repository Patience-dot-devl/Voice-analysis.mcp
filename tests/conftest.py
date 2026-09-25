import subprocess
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


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
def stereo_overlap(tmp_path_factory) -> str:
    """12s stereo file with overlap and a mid-turn pause.

    channel 0: speaks 0-4, pauses, 4.5-6 (one turn with an internal pause)
    channel 1: speaks 5-8 (interrupts channel 0 at 5s, overlaps 1s), then 9-9.15 (backchannel)
    channel 0: speaks 8.5-12 (responds 0.5s after channel 1 stops)
    """
    out = tmp_path_factory.mktemp("audio") / "stereo_overlap.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=220:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-filter_complex",
        "[0:a]volume='if(lt(t,4)+between(t,4.5,6)+gte(t,8.5),1,0)':eval=frame[a];"
        "[1:a]volume='if(between(t,5,8)+between(t,9,9.15),1,0)':eval=frame[b];"
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
def speech_wav() -> str:
    """Short synthesized English speech clip, 16 kHz mono, committed under tests/data."""
    return str(DATA_DIR / "speech_16k.wav")


@pytest.fixture(scope="session")
def narrowband_speech(tmp_path_factory, speech_wav) -> str:
    """The speech clip passed through 8 kHz and back up to 16 kHz (telephone bandwidth)."""
    d = tmp_path_factory.mktemp("audio")
    low = d / "speech_8k.wav"
    out = d / "speech_8k_up16k.wav"
    _ffmpeg("-i", speech_wav, "-ar", "8000", str(low))
    _ffmpeg("-i", str(low), "-ar", "16000", str(out))
    return str(out)
