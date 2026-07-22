"""Conversation-dynamics metrics: talk time, turns, pauses, overlap, interruptions.

Stereo call recordings (one speaker per channel — the usual telephony export
format) get full per-speaker metrics. Mono recordings get overall
speech/silence dynamics only, since energy-based VAD cannot tell speakers
apart; use transcription for who-said-what on mono audio.
"""

from __future__ import annotations

import numpy as np

from . import vad
from .audio_io import load_audio

ANALYSIS_SR = 16000
RESPONSE_LATENCY_MAX_S = 5.0


def analyze(path: str, start: float | None = None, end: float | None = None) -> dict:
    y, sr = load_audio(path, sample_rate=ANALYSIS_SR, mono=False, start=start, end=end)
    if y.ndim == 1:
        y = y[:, None]
    n_channels = y.shape[1]
    duration = y.shape[0] / sr

    per_channel = []
    masks = []
    hop_s = vad.HOP_MS / 1000
    for ch in range(n_channels):
        mono = np.ascontiguousarray(y[:, ch])
        mask, hop_s, threshold = vad.speech_mask(mono, sr)
        masks.append(mask)
        segs = [(s * hop_s, e * hop_s) for s, e in vad._runs(mask)]
        pauses = [b[0] - a[1] for a, b in zip(segs, segs[1:]) if b[0] - a[1] > 0.05]
        per_channel.append(
            {
                "channel": ch,
                "talk_time_seconds": round(sum(e - s for s, e in segs), 2),
                "talk_ratio": round(sum(e - s for s, e in segs) / duration, 3) if duration else 0.0,
                "turn_count": len(segs),
                "mean_turn_seconds": round(float(np.mean([e - s for s, e in segs])), 2) if segs else 0.0,
                "longest_turn_seconds": round(max((e - s for s, e in segs), default=0.0), 2),
                "longest_internal_pause_seconds": round(max(pauses, default=0.0), 2),
                "vad_threshold_dbfs": round(threshold, 1),
            }
        )

    n = min(m.size for m in masks)
    any_speech = np.zeros(n, dtype=bool)
    for m in masks:
        any_speech |= m[:n]
    silence_segs = [(s * hop_s, e * hop_s) for s, e in vad._runs(~any_speech)]
    dead_air = [seg for seg in silence_segs if seg[1] - seg[0] >= 1.0]

    result: dict = {
        "window": {"start_seconds": start or 0.0, "duration_seconds": round(duration, 2)},
        "channel_count": n_channels,
        "overall": {
            "speech_ratio": round(float(np.mean(any_speech)), 3),
            "dead_air_count_over_1s": len(dead_air),
            "longest_dead_air_seconds": round(max((e - s for s, e in dead_air), default=0.0), 2),
            "dead_air_over_3s": [
                {"start": round(s, 2), "end": round(e, 2)}
                for s, e in dead_air
                if e - s >= 3.0
            ][:20],
        },
        "per_channel": per_channel,
    }

    if n_channels >= 2:
        a, b = masks[0][:n], masks[1][:n]
        overlap = a & b
        result["speaker_interaction"] = {
            "note": "Assumes one speaker per channel (typical stereo call recording).",
            "overlap_seconds": round(float(np.sum(overlap)) * hop_s, 2),
            "overlap_ratio_of_speech": round(
                float(np.sum(overlap)) / max(1, int(np.sum(any_speech))), 3
            ),
            "interruptions_by_channel_0": _count_interruptions(b, a),
            "interruptions_by_channel_1": _count_interruptions(a, b),
            "response_latency": _response_latencies(a, b, hop_s),
        }
    else:
        result["note"] = (
            "Mono recording: speakers cannot be separated by channel. "
            "Use the transcribe tool for who-said-what and turn-taking."
        )
    return result


def _count_interruptions(active: np.ndarray, interrupter: np.ndarray) -> int:
    """Times `interrupter` starts speaking while `active` is already speaking."""
    starts = np.where(np.diff(interrupter.astype(np.int8)) == 1)[0] + 1
    return int(np.sum(active[starts]))


def _response_latencies(a: np.ndarray, b: np.ndarray, hop_s: float) -> dict:
    """Gaps between one side stopping and the other starting (both directions)."""
    latencies: list[float] = []
    for first, second in ((a, b), (b, a)):
        ends = np.where(np.diff(first.astype(np.int8)) == -1)[0] + 1
        starts = np.where(np.diff(second.astype(np.int8)) == 1)[0] + 1
        for e in ends:
            nxt = starts[starts >= e]
            if nxt.size:
                gap = (nxt[0] - e) * hop_s
                if 0 <= gap <= RESPONSE_LATENCY_MAX_S:
                    latencies.append(gap)
    if not latencies:
        return {"sample_count": 0}
    return {
        "sample_count": len(latencies),
        "median_seconds": round(float(np.median(latencies)), 2),
        "p90_seconds": round(float(np.percentile(latencies, 90)), 2),
    }
