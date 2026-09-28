"""Prosodic feature extraction: pitch, energy, pausing, delivery pace."""

from __future__ import annotations

import numpy as np

from . import vad
from .audio_io import AudioError, load_audio

MAX_WINDOW_S = 300.0
ANALYSIS_SR = 16000
PYIN_FRAME = 2048
PYIN_HOP = 512


def analyze(
    path: str,
    start: float | None = None,
    end: float | None = None,
    channel: int | None = None,
) -> dict:
    y, sr = load_audio(path, sample_rate=ANALYSIS_SR, mono=True, start=start, end=end, channel=channel)
    duration = y.size / sr
    if duration > MAX_WINDOW_S:
        raise AudioError(
            f"Prosody analysis window is {duration:.0f}s; max is {MAX_WINDOW_S:.0f}s. "
            "Pass start_time/end_time to analyze a shorter window (e.g. one speaker turn)."
        )

    import librosa  # heavy import, keep local

    rms_db, hop_s = vad.frame_rms_db(y, sr)
    segments, threshold_db = vad.speech_segments(y, sr)
    speech_time = sum(e - s for s, e in segments)
    pauses = vad.pauses_between(segments)

    return {
        "window": {
            "start_seconds": start or 0.0,
            "duration_seconds": round(duration, 2),
            "channel": channel,
        },
        "pitch": _pitch_stats(y, sr, librosa),
        "energy": {
            "rms_dbfs_mean": round(float(np.mean(rms_db)), 1),
            "rms_dbfs_p95": round(float(np.percentile(rms_db, 95)), 1),
            "dynamic_range_db": round(
                float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)), 1
            ),
        },
        "pace": _pace_stats(y, sr, segments, speech_time, librosa),
        "pausing": {
            "speech_time_seconds": round(speech_time, 2),
            "speech_ratio": round(speech_time / duration, 3) if duration else 0.0,
            "speech_segment_count": len(segments),
            "pause_count_over_300ms": sum(1 for p in pauses if p >= 0.3),
            "pause_count_over_1s": sum(1 for p in pauses if p >= 1.0),
            "longest_pause_seconds": round(max(pauses), 2) if pauses else 0.0,
            "mean_pause_seconds": round(float(np.mean(pauses)), 2) if pauses else 0.0,
            "vad_threshold_dbfs": round(threshold_db, 1),
        },
        "interpretation_hints": {
            "pitch_range_semitones": "under ~4 = monotone delivery; 6-12 = typical expressive speech",
            "voiced_ratio_of_speech": "share of speech frames with a detectable pitch (vowels); ~0.4-0.7 is normal",
            "syllables_per_second": "estimated from energy onsets; ~3-4 slow, 4-5 typical, 6+ fast",
            "speech_ratio": "share of the window with active speech (not silence)",
        },
    }


def _pitch_stats(y: np.ndarray, sr: int, librosa) -> dict:
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
        fmax=float(librosa.note_to_hz("C6")),  # ~1047 Hz
        sr=sr,
        frame_length=PYIN_FRAME,
        hop_length=PYIN_HOP,
    )
    voiced = np.asarray(voiced_flag, dtype=bool)
    voiced_f0 = f0[voiced & np.isfinite(f0)]

    # Restrict the voiced ratio to frames the VAD calls speech, so silence in
    # the window does not drag it down.
    speech_mask, hop_s, _ = vad.speech_mask(y, sr)
    pyin_times = librosa.frames_to_time(np.arange(voiced.size), sr=sr, hop_length=PYIN_HOP)
    speech_at_pyin = speech_mask[np.minimum((pyin_times / hop_s).astype(int), speech_mask.size - 1)]
    n_speech = int(np.sum(speech_at_pyin))

    pitch: dict = {
        "voiced_ratio_of_speech": round(float(np.sum(voiced & speech_at_pyin) / n_speech), 3)
        if n_speech
        else 0.0,
    }
    if voiced_f0.size >= 5:
        p5, p95 = np.percentile(voiced_f0, [5, 95])
        median = float(np.median(voiced_f0))
        pitch.update(
            {
                "median_hz": round(median, 1),
                "mean_hz": round(float(np.mean(voiced_f0)), 1),
                "p5_hz": round(float(p5), 1),
                "p95_hz": round(float(p95), 1),
                "range_semitones": round(float(12 * np.log2(p95 / p5)), 1),
                "std_semitones": round(float(np.std(12 * np.log2(voiced_f0 / median))), 2),
            }
        )
    else:
        pitch["note"] = "Too little voiced speech in this window for reliable pitch statistics."
    return pitch


def _pace_stats(
    y: np.ndarray, sr: int, segments: list[tuple[float, float]], speech_time: float, librosa
) -> dict:
    """Syllable-rate estimate from energy onsets in the speech band.

    Syllable nuclei show up as energy peaks in roughly 300-3000 Hz. Counting
    onsets there gives a rough articulation rate without needing a transcript.
    """
    if speech_time <= 0:
        return {"estimated_syllables_per_second": None, "note": "no speech detected"}
    hop = 160  # 10 ms
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=40, fmin=300, fmax=3000
    )
    strength = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max), sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=strength, sr=sr, hop_length=hop, units="time", backtrack=False,
        wait=6,  # ~60 ms minimum between syllables
    )
    in_speech = [t for t in onsets if any(s <= t <= e for s, e in segments)]
    return {
        "estimated_syllables_per_second": round(float(len(in_speech) / speech_time), 2),
        "estimated_syllable_count": len(in_speech),
        "mean_speech_segment_seconds": round(float(speech_time / len(segments)), 2) if segments else 0.0,
    }
