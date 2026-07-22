"""Lightweight energy-based voice activity detection shared by analysis modules."""

from __future__ import annotations

import numpy as np

FRAME_MS = 25
HOP_MS = 10
MIN_SPEECH_S = 0.15
MIN_GAP_S = 0.25


def frame_rms_db(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Per-frame RMS in dBFS. Returns (rms_db, hop_seconds)."""
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    n_frames = 1 + (y.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
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


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
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
    return list(zip(starts, ends))


def _close_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    out = mask.copy()
    for start, end in _runs(~mask):
        if start > 0 and end < mask.size and (end - start) <= max_gap_frames:
            out[start:end] = True
    return out


def _drop_short_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    out = mask.copy()
    for start, end in _runs(mask):
        if (end - start) < min_frames:
            out[start:end] = False
    return out


def speech_segments(y: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], float]:
    """Speech segments as (start_s, end_s) plus the detection threshold in dBFS."""
    mask, hop_s, threshold = speech_mask(y, sr)
    segs = [(round(s * hop_s, 3), round(e * hop_s, 3)) for s, e in _runs(mask)]
    return segs, threshold
