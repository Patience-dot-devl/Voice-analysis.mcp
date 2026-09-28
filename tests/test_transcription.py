"""Transcription tests run only when the tiny Whisper model is already cached,
so the suite never downloads models on its own."""

import pytest

from voice_analysis_mcp import audio_io, transcription


@pytest.fixture(scope="module")
def tiny_model():
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
    except Exception as exc:  # model not cached / backend missing
        pytest.skip(f"tiny Whisper model not available locally: {exc}")
    transcription._models["tiny"] = model
    return model


def test_transcribe_speech(tiny_model, speech_wav):
    r = transcription.transcribe(speech_wav, model_size="tiny", language="en", word_timestamps=True)
    assert r["language"] == "en"
    assert r["segment_count"] >= 1
    text = " ".join(s["text"] for s in r["segments"]).lower()
    assert "customer" in text or "support" in text
    assert r["words_per_minute"] and r["words_per_minute"] > 60
    assert all("words" in s for s in r["segments"])


def test_transcribe_window_offsets_timestamps(tiny_model, speech_wav):
    r = transcription.transcribe(speech_wav, model_size="tiny", language="en", start=2.0)
    assert all(s["start"] >= 2.0 for s in r["segments"])


def test_transcribe_split_channels(tiny_model, speech_wav, tmp_path):
    import subprocess

    stereo = tmp_path / "stereo.wav"
    # channel 0 = speech, channel 1 = silence
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", speech_wav,
         "-filter_complex", "[0:a]asplit[a][b];[b]volume=0[c];[a][c]join=inputs=2:channel_layout=stereo",
         str(stereo)],
        check=True, capture_output=True,
    )
    r = transcription.transcribe(str(stereo), model_size="tiny", language="en", split_channels=True)
    assert r["split_channels"] is True
    assert {s["speaker"] for s in r["segments"]} == {"channel_0"}
    starts = [s["start"] for s in r["segments"]]
    assert starts == sorted(starts)


def test_transcribe_truncates(tiny_model, speech_wav):
    r = transcription.transcribe(speech_wav, model_size="tiny", language="en", max_segments=1)
    assert r["segment_count"] == 1
    assert r["truncated"] is True
    assert r["next_start_time"] == r["segments"][0]["end"]


def test_transcribe_rejects_bad_args(speech_wav):
    with pytest.raises(audio_io.AudioError, match="model_size"):
        transcription.transcribe(speech_wav, model_size="huge")
    with pytest.raises(audio_io.AudioError, match="not both"):
        transcription.transcribe(speech_wav, model_size="tiny", channel=0, split_channels=True)
