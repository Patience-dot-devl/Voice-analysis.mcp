import numpy as np
import pytest

from voice_analysis_mcp import audio_io, conversation, prosody, quality, visuals


def test_audio_summary(stereo_call):
    meta = audio_io.audio_summary(stereo_call)
    assert meta["channels"] == 2
    assert meta["sample_rate_hz"] == 16000
    assert abs(meta["duration_seconds"] - 20.0) < 0.1


def test_load_audio_window_and_channel(stereo_call):
    y, sr = audio_io.load_audio(stereo_call, mono=False)
    assert y.ndim == 2 and y.shape[1] == 2

    # channel 1 is silent during 0-3s, channel 0 is active
    ch0, _ = audio_io.load_audio(stereo_call, channel=0, start=0.5, end=2.5)
    ch1, _ = audio_io.load_audio(stereo_call, channel=1, start=0.5, end=2.5)
    # lavfi sine defaults to amplitude 1/8, so active RMS is ~0.088
    assert float(np.sqrt(np.mean(ch0**2))) > 0.05
    assert float(np.sqrt(np.mean(ch1**2))) < 0.01


def test_load_audio_bad_channel(stereo_call):
    with pytest.raises(audio_io.AudioError, match="out of range"):
        audio_io.load_audio(stereo_call, channel=5)


def test_conversation_stereo(stereo_call):
    r = conversation.analyze(stereo_call)
    assert r["channel_count"] == 2
    ch0, ch1 = r["per_channel"]
    # channel 0 speaks 0-3,6-9,12-15,18-20 = 11s; channel 1 speaks 9s
    assert 9.5 < ch0["talk_time_seconds"] < 12.5
    assert 7.5 < ch1["talk_time_seconds"] < 10.5
    assert ch0["turn_count"] == 4
    assert ch1["turn_count"] == 3
    assert "speaker_interaction" in r
    assert r["speaker_interaction"]["overlap_seconds"] < 1.0


def test_conversation_dead_air(tone_with_silence):
    r = conversation.analyze(tone_with_silence)
    assert r["channel_count"] == 1
    assert r["overall"]["dead_air_count_over_1s"] == 1
    assert 2.0 < r["overall"]["longest_dead_air_seconds"] < 4.0
    assert "note" in r  # mono note about speaker separation


def test_quality_flags_silence_gap_snr(tone_with_silence):
    r = quality.analyze(tone_with_silence)
    assert r["levels"]["clipped_sample_ratio"] == 0
    assert r["noise"]["estimated_snr_db"] is None or r["noise"]["estimated_snr_db"] > 20


def test_quality_detects_clipping(tmp_path, tone_with_silence):
    import subprocess

    clipped = tmp_path / "clipped.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", tone_with_silence,
         "-af", "volume=20", str(clipped)],
        check=True, capture_output=True,
    )
    r = quality.analyze(str(clipped))
    assert any("clipping" in i for i in r["issues"])


def test_prosody_on_speech(speech_wav):
    r = prosody.analyze(speech_wav)
    assert 80 < r["pitch"]["median_hz"] < 400
    assert r["pausing"]["speech_ratio"] > 0.5


def test_prosody_window_limit(tone_with_silence, monkeypatch):
    monkeypatch.setattr(prosody, "MAX_WINDOW_S", 5.0)
    with pytest.raises(audio_io.AudioError, match="max is 5s"):
        prosody.analyze(tone_with_silence)


def test_extract_segment(tone_with_silence, tmp_path):
    out = audio_io.extract_segment_to_file(tone_with_silence, 1.0, 3.5, str(tmp_path / "seg.wav"))
    assert abs(audio_io.audio_summary(out)["duration_seconds"] - 2.5) < 0.1


def test_visuals_produce_png(stereo_call):
    spec = visuals.spectrogram_png(stereo_call, start=0, end=5)
    wave = visuals.waveform_png(stereo_call)
    assert spec[:8] == b"\x89PNG\r\n\x1a\n"
    assert wave[:8] == b"\x89PNG\r\n\x1a\n"
