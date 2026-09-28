"""Lightweight energy-based voice activity detection shared by analysis modules."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FRAME_MS = 25
HOP_MS = 10
MIN_SPEECH_S = 0.15
MIN_GAP_S = 0.25
MIN_PAUSE_S = 0.05


def frame_rms_db(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Per-frame RMS in dBFS. Returns (rms_db, hop_seconds).

    Uses a zero-copy sliding window so memory stays O(n) even for hour-long
    recordings.
    """
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    y = np.asarray(y, dtype=np.float32)
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    windows = sliding_window_view(y, frame)[::hop]
    # einsum computes sum of squares per row without materialising y**2
    power = np.einsum("ij,ij->i", windows, windows, dtype=np.float64) / frame
    rms = np.sqrt(power + 1e-12)
    return 20 * np.log10(rms + 1e-12), hop / sr


def speech_mask(y: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """Boolean speech mask per frame.

    Threshold adapts to the recording's noise floor.
    Returns (mask, hop_seconds, threshold_db).
    """
    rms_db, hop_s = frame_rms_db(y, sr)
    noise_floor = float(np.percentile(rms_db, 10))
    threshold = max(noise_floor + 9.0, -55.0)
    mask = rms_db > threshold

    mask = _close_gaps(mask, int(round(MIN_GAP_S / hop_s)))
    mask = _drop_short_runs(mask, int(round(MIN_SPEECH_S / hop_s)))
    return mask, hop_s, threshold


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of contiguous True runs; end is exclusive."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends, strict=True))


def _close_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    out = mask.copy()
    for start, end in runs(~mask):
        if start > 0 and end < mask.size and (end - start) <= max_gap_frames:
            out[start:end] = True
    return out


def _drop_short_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    out = mask.copy()
    for start, end in runs(mask):
        if (end - start) < min_frames:
            out[start:end] = False
    return out


def mask_to_segments(mask: np.ndarray, hop_s: float) -> list[tuple[float, float]]:
    """Convert a frame mask to (start_s, end_s) segments."""
    return [(s * hop_s, e * hop_s) for s, e in runs(mask)]


def speech_segments(y: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], float]:
    """Speech segments as (start_s, end_s) plus the detection threshold in dBFS."""
    mask, hop_s, threshold = speech_mask(y, sr)
    segs = [(round(s, 3), round(e, 3)) for s, e in mask_to_segments(mask, hop_s)]
    return segs, threshold


def pauses_between(segments: list[tuple[float, float]]) -> list[float]:
    """Gaps (seconds) between consecutive speech segments, ignoring tiny ones."""
    return [
        round(nxt[0] - cur[1], 3)
        for cur, nxt in zip(segments, segments[1:], strict=False)
        if nxt[0] - cur[1] > MIN_PAUSE_S
    ]
