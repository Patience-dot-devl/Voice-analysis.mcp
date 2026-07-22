"""Prosodic feature extraction: pitch, energy, pausing, delivery pace."""

from __future__ import annotations

import numpy as np

from . import vad
from .audio_io import AudioError, load_audio

MAX_WINDOW_S = 300.0
ANALYSIS_SR = 16000


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

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
        fmax=float(librosa.note_to_hz("C6")),  # ~1047 Hz
        sr=sr,
        frame_length=2048,
    )
    voiced_f0 = f0[np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0)]

    pitch: dict = {"voiced_ratio": round(float(np.mean(np.asarray(voiced_flag, dtype=bool))), 3)}
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
                "std_semitones": round(
                    float(np.std(12 * np.log2(voiced_f0 / median))), 2
                ),
            }
        )
    else:
        pitch["note"] = "Too little voiced speech in this window for reliable pitch statistics."

    rms_db, _hop_s = vad.frame_rms_db(y, sr)
    segments, threshold_db = vad.speech_segments(y, sr)
    speech_time = sum(e - s for s, e in segments)
    pauses = _pauses_between(segments)

    energy = {
        "rms_dbfs_mean": round(float(np.mean(rms_db)), 1),
        "rms_dbfs_p95": round(float(np.percentile(rms_db, 95)), 1),
        "dynamic_range_db": round(float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)), 1),
    }

    pausing = {
        "speech_time_seconds": round(speech_time, 2),
        "speech_ratio": round(speech_time / duration, 3) if duration else 0.0,
        "speech_segment_count": len(segments),
        "pause_count_over_300ms": sum(1 for p in pauses if p >= 0.3),
        "pause_count_over_1s": sum(1 for p in pauses if p >= 1.0),
        "longest_pause_seconds": round(max(pauses), 2) if pauses else 0.0,
        "mean_pause_seconds": round(float(np.mean(pauses)), 2) if pauses else 0.0,
        "vad_threshold_dbfs": round(threshold_db, 1),
    }

    return {
        "window": {
            "start_seconds": start or 0.0,
            "duration_seconds": round(duration, 2),
            "channel": channel,
        },
        "pitch": pitch,
        "energy": energy,
        "pausing": pausing,
        "interpretation_hints": {
            "pitch_range_semitones": "under ~4 = monotone delivery; 6-12 = typical expressive speech",
            "speech_ratio": "share of the window with active speech (not silence)",
        },
    }


def _pauses_between(segments: list[tuple[float, float]]) -> list[float]:
    return [
        round(nxt[0] - cur[1], 3)
        for cur, nxt in zip(segments, segments[1:])
        if nxt[0] - cur[1] > 0.05
    ]
