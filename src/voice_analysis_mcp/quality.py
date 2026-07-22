"""Recording-quality metrics: levels, clipping, noise, estimated SNR."""

from __future__ import annotations

import numpy as np

from . import vad
from .audio_io import audio_summary, load_audio

ANALYSIS_SR = 16000
CLIP_THRESHOLD = 0.999


def analyze(path: str, start: float | None = None, end: float | None = None) -> dict:
    meta = audio_summary(path)
    y, sr = load_audio(path, sample_rate=ANALYSIS_SR, mono=True, start=start, end=end)
    duration = y.size / sr

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    clipped = int(np.sum(np.abs(y) >= CLIP_THRESHOLD))
    dc_offset = float(np.mean(y))

    mask, _hop_s, threshold = vad.speech_mask(y, sr)
    rms_db, _ = vad.frame_rms_db(y, sr)
    n = min(mask.size, rms_db.size)
    speech_db = rms_db[:n][mask[:n]]
    noise_db = rms_db[:n][~mask[:n]]

    snr_db = None
    if speech_db.size > 10 and noise_db.size > 10:
        snr_db = round(float(np.median(speech_db) - np.median(noise_db)), 1)

    issues: list[str] = []
    if clipped > y.size * 1e-5:
        issues.append("clipping detected — audio distorts at peaks")
    if 20 * np.log10(rms + 1e-12) < -35:
        issues.append("very low recording level")
    if snr_db is not None and snr_db < 15:
        issues.append("high background noise relative to speech")
    if abs(dc_offset) > 0.01:
        issues.append("significant DC offset")
    if meta["sample_rate_hz"] and meta["sample_rate_hz"] <= 8000:
        issues.append("narrowband (telephone-quality) audio — 8 kHz sample rate")

    return {
        "window": {"start_seconds": start or 0.0, "duration_seconds": round(duration, 2)},
        "source": {
            "codec": meta["codec"],
            "sample_rate_hz": meta["sample_rate_hz"],
            "channels": meta["channels"],
            "bit_rate": meta["bit_rate"],
        },
        "levels": {
            "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
            "rms_dbfs": round(20 * np.log10(rms + 1e-12), 1),
            "clipped_sample_ratio": round(clipped / y.size, 6),
            "dc_offset": round(dc_offset, 4),
        },
        "noise": {
            "estimated_snr_db": snr_db,
            "noise_floor_dbfs": round(float(np.median(noise_db)), 1) if noise_db.size else None,
            "vad_threshold_dbfs": round(threshold, 1),
        },
        "issues": issues or ["no obvious quality issues detected"],
    }
