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
# An onset overlapping the other speaker for at least this long counts as an
# interruption; shorter overlaps are reported separately as backchannels
# ("mm-hm", "right") which are not attempts to take the floor.
MIN_INTERRUPTION_OVERLAP_S = 0.3
# Overlaps shorter than this are hand-off jitter, not a backchannel.
MIN_BACKCHANNEL_OVERLAP_S = 0.1
DEAD_AIR_MIN_S = 1.0
DEAD_AIR_REPORT_MIN_S = 3.0
DEAD_AIR_REPORT_MAX = 20


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
        mask, hop_s, threshold = vad.speech_mask(np.ascontiguousarray(y[:, ch]), sr)
        masks.append(mask)
        segs = vad.mask_to_segments(mask, hop_s)
        lengths = [e - s for s, e in segs]
        pauses = vad.pauses_between(segs)
        talk_time = sum(lengths)
        per_channel.append(
            {
                "channel": ch,
                "talk_time_seconds": round(talk_time, 2),
                "talk_ratio": round(talk_time / duration, 3) if duration else 0.0,
                "turn_count": len(segs),
                "mean_turn_seconds": round(float(np.mean(lengths)), 2) if lengths else 0.0,
                "longest_turn_seconds": round(max(lengths, default=0.0), 2),
                "longest_internal_pause_seconds": round(max(pauses, default=0.0), 2),
                "vad_threshold_dbfs": round(threshold, 1),
            }
        )

    any_speech = np.logical_or.reduce(masks)
    dead_air = [
        seg for seg in vad.mask_to_segments(~any_speech, hop_s)
        if seg[1] - seg[0] >= DEAD_AIR_MIN_S
    ]
    window_start = start or 0.0

    result: dict = {
        "window": {"start_seconds": window_start, "duration_seconds": round(duration, 2)},
        "channel_count": n_channels,
        "overall": {
            "speech_ratio": round(float(np.mean(any_speech)), 3),
            "dead_air_count_over_1s": len(dead_air),
            "longest_dead_air_seconds": round(max((e - s for s, e in dead_air), default=0.0), 2),
            "dead_air_over_3s": [
                {"start": round(window_start + s, 2), "end": round(window_start + e, 2)}
                for s, e in dead_air
                if e - s >= DEAD_AIR_REPORT_MIN_S
            ][:DEAD_AIR_REPORT_MAX],
        },
        "per_channel": per_channel,
    }

    if n_channels >= 2:
        a, b = masks[0], masks[1]
        overlap = a & b
        by_0 = _classify_overlaps(active=b, interrupter=a, hop_s=hop_s)
        by_1 = _classify_overlaps(active=a, interrupter=b, hop_s=hop_s)
        result["speaker_interaction"] = {
            "note": (
                "Assumes one speaker per channel (typical stereo call recording). "
                "Interruptions are onsets that overlap the other speaker for >= "
                f"{MIN_INTERRUPTION_OVERLAP_S}s; shorter overlaps are counted as backchannels."
            ),
            "overlap_seconds": round(float(np.sum(overlap)) * hop_s, 2),
            "overlap_ratio_of_speech": round(
                float(np.sum(overlap)) / max(1, int(np.sum(any_speech))), 3
            ),
            "interruptions_by_channel_0": by_0["interruptions"],
            "interruptions_by_channel_1": by_1["interruptions"],
            "backchannels_by_channel_0": by_0["backchannels"],
            "backchannels_by_channel_1": by_1["backchannels"],
            "response_latency": _response_latencies(a, b, hop_s),
        }
        if n_channels > 2:
            result["speaker_interaction"]["note"] += (
                f" File has {n_channels} channels; only channels 0 and 1 are compared."
            )
    else:
        result["note"] = (
            "Mono recording: speakers cannot be separated by channel. "
            "Use the transcribe tool for who-said-what and turn-taking."
        )
    return result


def _onsets(mask: np.ndarray) -> np.ndarray:
    """Frame indices where `mask` switches from False to True."""
    return np.where(np.diff(mask.astype(np.int8)) == 1)[0] + 1


def _offsets(mask: np.ndarray) -> np.ndarray:
    """Frame indices where `mask` switches from True to False."""
    return np.where(np.diff(mask.astype(np.int8)) == -1)[0] + 1


def _classify_overlaps(active: np.ndarray, interrupter: np.ndarray, hop_s: float) -> dict:
    """Split `interrupter` onsets during `active` speech into interruptions and backchannels."""
    both = active & interrupter
    interruptions = backchannels = 0
    for i in _onsets(interrupter):
        if not active[i]:
            continue
        # length of the joint-speech run starting at this onset
        j = i
        while j < both.size and both[j]:
            j += 1
        overlap_s = (j - i) * hop_s
        if overlap_s >= MIN_INTERRUPTION_OVERLAP_S:
            interruptions += 1
        elif overlap_s >= MIN_BACKCHANNEL_OVERLAP_S:
            backchannels += 1
    return {"interruptions": interruptions, "backchannels": backchannels}


def _response_latencies(a: np.ndarray, b: np.ndarray, hop_s: float) -> dict:
    """Gaps between one side finishing a turn and the other starting (both directions).

    A "finish" only counts if the same side does not resume before the other
    side starts, so a speaker pausing mid-turn does not produce a sample. A
    tight hand-off where the reply starts slightly before the first speaker
    stops (less than the interruption threshold) counts as zero latency.
    """
    latencies: list[float] = []
    lead_frames = int(round(MIN_INTERRUPTION_OVERLAP_S / hop_s))
    for first, second in ((a, b), (b, a)):
        ends = _offsets(first)
        restarts = _onsets(first)
        starts = _onsets(second)
        for e in ends:
            nxt = starts[starts >= e - lead_frames]
            if not nxt.size:
                continue
            resumed_first = restarts[(restarts > e) & (restarts < nxt[0])]
            if resumed_first.size:
                continue
            gap = max(0.0, (nxt[0] - e) * hop_s)
            if gap <= RESPONSE_LATENCY_MAX_S:
                latencies.append(gap)
    if not latencies:
        return {"sample_count": 0}
    return {
        "sample_count": len(latencies),
        "median_seconds": round(float(np.median(latencies)), 2),
        "p90_seconds": round(float(np.percentile(latencies, 90)), 2),
    }
