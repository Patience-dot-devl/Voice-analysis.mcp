import subprocess

import numpy as np
import pytest

from voice_analysis_mcp import audio_io, conversation, prosody, quality, vad, visuals


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


@pytest.mark.parametrize(
    "start, end, match",
    [
        (-5, 5, "start_time must be >= 0"),
        (5, 5, "must be greater"),
        (6, 2, "must be greater"),
        (25, None, "beyond the end"),
        (0, 30, "beyond the end"),
    ],
)
def test_load_audio_rejects_bad_windows(stereo_call, start, end, match):
    with pytest.raises(audio_io.AudioError, match=match):
        audio_io.load_audio(stereo_call, start=start, end=end)


def test_load_audio_tolerates_end_at_duration(stereo_call):
    y, sr = audio_io.load_audio(stereo_call, start=19, end=20.0)
    assert abs(y.size / sr - 1.0) < 0.05


def test_probe_is_cached(stereo_call, monkeypatch):
    calls = []
    real = audio_io.subprocess.run

    def spy(cmd, *a, **kw):
        if cmd[0] == "ffprobe":
            calls.append(cmd)
        return real(cmd, *a, **kw)

    monkeypatch.setattr(audio_io.subprocess, "run", spy)
    audio_io._probe_cached.cache_clear()
    quality.analyze(stereo_call)  # needs metadata and samples
    assert len(calls) == 1


def test_frame_rms_db_matches_naive_and_stays_small():
    sr = 16000
    y = (np.random.default_rng(0).standard_normal(sr * 20) * 0.1).astype(np.float32)
    db, hop_s = vad.frame_rms_db(y, sr)
    frame, hop = 400, 160
    n = 1 + (y.size - frame) // hop
    naive = np.array([np.sqrt(np.mean(y[i * hop : i * hop + frame] ** 2)) for i in range(n)])
    assert db.size == n
    assert hop_s == pytest.approx(0.01)
    np.testing.assert_allclose(db, 20 * np.log10(naive + 1e-12), atol=1e-3)


def test_conversation_stereo(stereo_call):
    r = conversation.analyze(stereo_call)
    assert r["channel_count"] == 2
    ch0, ch1 = r["per_channel"]
    # channel 0 speaks 0-3,6-9,12-15,18-20 = 11s; channel 1 speaks 9s
    assert 9.5 < ch0["talk_time_seconds"] < 12.5
    assert 7.5 < ch1["talk_time_seconds"] < 10.5
    assert ch0["turn_count"] == 4
    assert ch1["turn_count"] == 3
    si = r["speaker_interaction"]
    assert si["overlap_seconds"] < 1.0
    assert si["interruptions_by_channel_0"] == 0
    assert si["interruptions_by_channel_1"] == 0
    # clean hand-offs every 3s: latency ~0
    assert si["response_latency"]["sample_count"] == 6
    assert si["response_latency"]["median_seconds"] < 0.2


def test_conversation_interruptions_and_latency(stereo_overlap):
    si = conversation.analyze(stereo_overlap)["speaker_interaction"]
    # channel 1 starts at 5s while channel 0 is talking (4.5-6): one interruption
    assert si["interruptions_by_channel_1"] == 1
    # the 150 ms blip at 9s lands inside channel 0's turn -> backchannel
    assert si["backchannels_by_channel_1"] == 1
    assert si["interruptions_by_channel_0"] == 0
    assert 0.8 < si["overlap_seconds"] < 1.5
    # Latency samples: ch0 stops at 6 while ch1 already talking (not a response);
    # ch1 stops at 8, ch0 answers at 8.5 -> 0.5s. The ch0 pause at 4-4.5 must
    # NOT count: ch0 resumed before ch1 started.
    lat = si["response_latency"]
    assert lat["sample_count"] == 1
    assert 0.35 < lat["median_seconds"] < 0.65


def test_response_latency_ignores_mid_turn_pause():
    # A speaks 0-1s, pauses, 1.5-2s; B starts at 2.5s. Only one sample (0.5s).
    a = np.zeros(400, bool)
    a[0:100] = True
    a[150:200] = True
    b = np.zeros(400, bool)
    b[250:400] = True
    r = conversation._response_latencies(a, b, 0.01)
    assert r["sample_count"] == 1
    assert r["median_seconds"] == pytest.approx(0.5)


def test_classify_overlaps_splits_backchannels():
    active = np.zeros(300, bool)
    active[0:300] = True
    interrupter = np.zeros(300, bool)
    interrupter[50:60] = True  # 100 ms blip -> backchannel
    interrupter[100:200] = True  # 1 s -> interruption
    r = conversation._classify_overlaps(active, interrupter, 0.01)
    assert r == {"interruptions": 1, "backchannels": 1}


def test_conversation_dead_air(tone_with_silence):
    r = conversation.analyze(tone_with_silence)
    assert r["channel_count"] == 1
    assert r["overall"]["dead_air_count_over_1s"] == 1
    assert 2.0 < r["overall"]["longest_dead_air_seconds"] < 4.0
    assert "note" in r  # mono note about speaker separation


def test_conversation_dead_air_absolute_times(tone_with_silence, monkeypatch):
    monkeypatch.setattr(conversation, "DEAD_AIR_REPORT_MIN_S", 2.0)
    r = conversation.analyze(tone_with_silence, start=2.0)
    (gap,) = r["overall"]["dead_air_over_3s"]
    assert 3.5 < gap["start"] < 4.5
    assert 6.5 < gap["end"] < 7.5


def test_quality_snr_on_tone_with_silence(tone_with_silence):
    r = quality.analyze(tone_with_silence)
    assert r["levels"]["clipped_sample_ratio"] == 0
    assert r["noise"]["estimated_snr_db"] is not None
    assert r["noise"]["estimated_snr_db"] > 20
    assert len(r["levels"]["per_channel"]) == 1


def test_quality_detects_clipping(tmp_path, tone_with_silence):
    clipped = tmp_path / "clipped.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", tone_with_silence,
         "-af", "volume=20", str(clipped)],
        check=True, capture_output=True,
    )
    r = quality.analyze(str(clipped))
    assert any("clipping" in i for i in r["issues"])


def test_quality_narrowband_detection(speech_wav, narrowband_speech):
    wide = quality.analyze(speech_wav)
    narrow = quality.analyze(narrowband_speech)
    assert wide["source"]["sample_rate_hz"] == narrow["source"]["sample_rate_hz"] == 16000
    assert not any("narrowband" in i for i in wide["issues"])
    assert any("narrowband" in i for i in narrow["issues"])
    assert narrow["bandwidth"]["energy_ratio_above_4khz"] < wide["bandwidth"]["energy_ratio_above_4khz"]


def test_quality_channel_imbalance(tmp_path, speech_wav):
    out = tmp_path / "imbalanced.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", speech_wav,
         "-filter_complex", "[0:a]asplit[a][b];[b]volume=0.05[c];[a][c]join=inputs=2:channel_layout=stereo",
         str(out)],
        check=True, capture_output=True,
    )
    r = quality.analyze(str(out))
    assert len(r["levels"]["per_channel"]) == 2
    assert any("imbalance" in i and "channel 1" in i for i in r["issues"])


def test_prosody_on_speech(speech_wav):
    r = prosody.analyze(speech_wav)
    assert 80 < r["pitch"]["median_hz"] < 400
    assert r["pitch"]["range_semitones"] > 4  # TTS voice is not monotone
    assert 0.3 < r["pitch"]["voiced_ratio_of_speech"] <= 1.0
    assert r["pausing"]["speech_ratio"] > 0.5
    assert 2.0 < r["pace"]["estimated_syllables_per_second"] < 7.0


def test_prosody_window_limit(tone_with_silence, monkeypatch):
    monkeypatch.setattr(prosody, "MAX_WINDOW_S", 5.0)
    with pytest.raises(audio_io.AudioError, match="max is 5s"):
        prosody.analyze(tone_with_silence)


def test_extract_segment(tone_with_silence, tmp_path):
    out = audio_io.extract_segment_to_file(tone_with_silence, 1.0, 3.5, str(tmp_path / "seg.wav"))
    assert abs(audio_io.audio_summary(out)["duration_seconds"] - 2.5) < 0.1


def test_extract_segment_refuses_overwrite(tone_with_silence, tmp_path):
    target = str(tmp_path / "seg.wav")
    audio_io.extract_segment_to_file(tone_with_silence, 1.0, 2.0, target)
    with pytest.raises(audio_io.AudioError, match="already exists"):
        audio_io.extract_segment_to_file(tone_with_silence, 1.0, 2.0, target)
    audio_io.extract_segment_to_file(tone_with_silence, 1.0, 3.0, target, overwrite=True)
    assert abs(audio_io.audio_summary(target)["duration_seconds"] - 2.0) < 0.1


def test_extract_segment_rejects_source_and_out_of_range(tone_with_silence):
    with pytest.raises(audio_io.AudioError, match="must differ"):
        audio_io.extract_segment_to_file(tone_with_silence, 1.0, 2.0, tone_with_silence, overwrite=True)
    with pytest.raises(audio_io.AudioError, match="beyond the end"):
        audio_io.extract_segment_to_file(tone_with_silence, 1.0, 20.0)


def test_visuals_produce_png(stereo_call):
    spec = visuals.spectrogram_png(stereo_call, start=0, end=5)
    wave = visuals.waveform_png(stereo_call)
    assert spec[:8] == b"\x89PNG\r\n\x1a\n"
    assert wave[:8] == b"\x89PNG\r\n\x1a\n"
