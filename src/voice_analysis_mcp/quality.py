"""Recording-quality metrics: levels, clipping, noise, estimated SNR, bandwidth."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import vad
from .audio_io import audio_summary, load_audio

ANALYSIS_SR = 16000
CLIP_THRESHOLD = 0.999
CLIP_RATIO_FLAG = 1e-5
LOW_LEVEL_DBFS = -35.0
LOW_SNR_DB = 15.0
DC_OFFSET_FLAG = 0.01
NARROWBAND_HZ = 4000.0  # telephone audio has nothing above ~3.4 kHz
# Wideband speech keeps ~0.5-5% of its energy above 4 kHz (fricatives); an
# upsampled telephone recording keeps ~1e-5. Flag below -30 dB.
NARROWBAND_ENERGY_RATIO = 1e-3
CHANNEL_IMBALANCE_DB = 12.0
SPECTRUM_FRAMES = 2000
SPECTRUM_NFFT = 1024


def analyze(path: str, start: float | None = None, end: float | None = None) -> dict:
    meta = audio_summary(path)
    y_multi, sr = load_audio(path, sample_rate=ANALYSIS_SR, mono=False, start=start, end=end)
    if y_multi.ndim == 1:
        y_multi = y_multi[:, None]
    n_channels = y_multi.shape[1]
    y = np.ascontiguousarray(y_multi.mean(axis=1)) if n_channels > 1 else y_multi[:, 0]
    duration = y.size / sr

    peak = float(np.max(np.abs(y_multi)))
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    rms_dbfs = _db(rms)
    clipped = int(np.sum(np.abs(y_multi) >= CLIP_THRESHOLD))
    clipped_ratio = clipped / y_multi.size
    dc_offset = float(np.mean(y))

    mask, _hop_s, threshold = vad.speech_mask(y, sr)
    rms_db, _ = vad.frame_rms_db(y, sr)
    speech_db = rms_db[mask]
    noise_db = rms_db[~mask]

    snr_db = None
    if speech_db.size > 10 and noise_db.size > 10:
        snr_db = round(float(np.median(speech_db) - np.median(noise_db)), 1)

    bandwidth = _bandwidth(y, sr, mask)

    per_channel = [
        {
            "channel": ch,
            "peak_dbfs": round(_db(float(np.max(np.abs(y_multi[:, ch])))), 1),
            "rms_dbfs": round(_db(float(np.sqrt(np.mean(y_multi[:, ch] ** 2) + 1e-12))), 1),
        }
        for ch in range(n_channels)
    ]

    issues: list[str] = []
    if clipped_ratio > CLIP_RATIO_FLAG:
        issues.append("clipping detected — audio distorts at peaks")
    if rms_dbfs < LOW_LEVEL_DBFS:
        issues.append("very low recording level")
    if snr_db is not None and snr_db < LOW_SNR_DB:
        issues.append("high background noise relative to speech")
    if abs(dc_offset) > DC_OFFSET_FLAG:
        issues.append("significant DC offset")
    if meta["sample_rate_hz"] and meta["sample_rate_hz"] <= 8000:
        issues.append("narrowband (telephone-quality) audio — 8 kHz sample rate")
    elif (
        bandwidth["energy_ratio_above_4khz"] is not None
        and bandwidth["energy_ratio_above_4khz"] < NARROWBAND_ENERGY_RATIO
    ):
        issues.append(
            "narrowband (telephone-quality) audio — almost no energy above 4 kHz "
            f"despite {meta['sample_rate_hz']} Hz sample rate (likely upsampled telephone audio)"
        )
    if n_channels > 1:
        levels = [c["rms_dbfs"] for c in per_channel]
        if max(levels) - min(levels) > CHANNEL_IMBALANCE_DB:
            quiet = int(np.argmin(levels))
            issues.append(
                f"channel level imbalance — channel {quiet} is {max(levels) - min(levels):.0f} dB quieter"
            )

    return {
        "window": {"start_seconds": start or 0.0, "duration_seconds": round(duration, 2)},
        "source": {
            "codec": meta["codec"],
            "sample_rate_hz": meta["sample_rate_hz"],
            "channels": meta["channels"],
            "bit_rate": meta["bit_rate"],
        },
        "levels": {
            "peak_dbfs": round(_db(peak), 1),
            "rms_dbfs": round(rms_dbfs, 1),
            "clipped_sample_ratio": round(clipped_ratio, 6),
            "dc_offset": round(dc_offset, 4),
            "per_channel": per_channel,
        },
        "noise": {
            "estimated_snr_db": snr_db,
            "noise_floor_dbfs": round(float(np.median(noise_db)), 1) if noise_db.size else None,
            "vad_threshold_dbfs": round(threshold, 1),
        },
        "bandwidth": bandwidth,
        "issues": issues or ["no obvious quality issues detected"],
    }


def _db(x: float) -> float:
    return float(20 * np.log10(x + 1e-12))


def _bandwidth(y: np.ndarray, sr: int, speech_mask: np.ndarray) -> dict:
    """Average power spectrum over (a subsample of) speech frames.

    Reports the 99% energy rolloff and the share of energy above 4 kHz; the
    latter separates upsampled telephone audio from genuine wideband speech.
    """
    hop = max(1, int(sr * vad.HOP_MS / 1000))
    if y.size < SPECTRUM_NFFT:
        return {"effective_bandwidth_hz": None, "energy_ratio_above_4khz": None}
    frames = sliding_window_view(y, SPECTRUM_NFFT)[::hop]
    n = min(frames.shape[0], speech_mask.size)
    idx = np.where(speech_mask[:n])[0]
    if idx.size < 10:
        idx = np.arange(n)
    if idx.size > SPECTRUM_FRAMES:
        idx = idx[np.linspace(0, idx.size - 1, SPECTRUM_FRAMES).astype(int)]
    window = np.hanning(SPECTRUM_NFFT).astype(np.float32)
    power = np.mean(np.abs(np.fft.rfft(frames[idx] * window, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(SPECTRUM_NFFT, 1 / sr)
    total = float(np.sum(power))
    if total <= 0:
        return {"effective_bandwidth_hz": None, "energy_ratio_above_4khz": None}
    cumulative = np.cumsum(power) / total
    rolloff = float(freqs[int(np.searchsorted(cumulative, 0.99))])
    above = float(np.sum(power[freqs >= NARROWBAND_HZ]) / total)
    return {
        "effective_bandwidth_hz": round(rolloff, 0),
        "energy_ratio_above_4khz": round(above, 5),
    }
