"""Audio decoding and metadata via ffmpeg/ffprobe.

All decoding goes through ffmpeg so any container/codec ffmpeg understands
(wav, mp3, m4a, ogg, flac, opus, telephony formats, video files, ...) works.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


class AudioError(RuntimeError):
    pass


def _require_ffmpeg() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise AudioError(
                f"'{exe}' not found on PATH. Install ffmpeg (e.g. `brew install ffmpeg`)."
            )


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise AudioError(f"File not found: {p}")
    if not p.is_file():
        raise AudioError(f"Not a file: {p}")
    return p


def probe(path: str) -> dict:
    """Return raw ffprobe metadata (format + streams) for a media file."""
    p = resolve_path(path)
    st = p.stat()
    return _probe_cached(str(p), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=64)
def _probe_cached(path: str, _mtime_ns: int, _size: int) -> dict:
    """ffprobe is a subprocess spawn; cache per (path, mtime, size) so one tool
    call that needs both metadata and samples probes only once."""
    _require_ffmpeg()
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def audio_summary(path: str) -> dict:
    """Condensed, LLM-friendly metadata for the first audio stream."""
    info = probe(path)
    fmt = info.get("format", {})
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise AudioError(f"No audio stream found in {path}")
    a = audio_streams[0]
    duration = float(fmt.get("duration") or a.get("duration") or 0.0)
    return {
        "path": str(resolve_path(path)),
        "format": fmt.get("format_name"),
        "codec": a.get("codec_name"),
        "duration_seconds": round(duration, 3),
        "sample_rate_hz": int(a.get("sample_rate") or 0),
        "channels": int(a.get("channels") or 0),
        "channel_layout": a.get("channel_layout"),
        "bit_rate": int(fmt.get("bit_rate") or 0) or None,
        "size_bytes": int(fmt.get("size") or 0) or None,
        "audio_stream_count": len(audio_streams),
        "tags": fmt.get("tags") or {},
    }


def validate_window(
    start: float | None, end: float | None, duration: float | None
) -> tuple[float, float | None]:
    """Normalise a [start, end] window against the file duration.

    Returns (start, end) with start >= 0. Raises AudioError for negative,
    inverted, or out-of-file windows. A missing duration (some streams don't
    report one) skips the upper-bound check.
    """
    s = 0.0 if start is None else float(start)
    e = None if end is None else float(end)
    if s < 0:
        raise AudioError(f"start_time must be >= 0, got {s}")
    if e is not None and e <= s:
        raise AudioError(f"end_time ({e}) must be greater than start_time ({s})")
    if duration:
        if s >= duration:
            raise AudioError(
                f"start_time ({s}) is beyond the end of the file ({duration:.2f}s)"
            )
        if e is not None and e > duration + 0.05:
            raise AudioError(
                f"end_time ({e}) is beyond the end of the file ({duration:.2f}s)"
            )
    return s, e


def load_audio(
    path: str,
    sample_rate: int | None = None,
    mono: bool = True,
    start: float | None = None,
    end: float | None = None,
    channel: int | None = None,
) -> tuple[np.ndarray, int]:
    """Decode audio to float32 numpy via ffmpeg.

    Returns (samples, sample_rate). samples is 1-D if mono/channel-selected,
    else shaped (n_frames, n_channels).
    """
    _require_ffmpeg()
    p = resolve_path(path)
    meta = audio_summary(path)
    src_channels = meta["channels"]
    sr = sample_rate or meta["sample_rate_hz"] or 16000
    start, end = validate_window(start, end, meta["duration_seconds"])

    if channel is not None:
        if not 0 <= channel < src_channels:
            raise AudioError(
                f"channel {channel} out of range: file has {src_channels} channel(s) (0-indexed)"
            )

    cmd = ["ffmpeg", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(p)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]

    if channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{channel}"]
        out_channels = 1
    elif mono:
        cmd += ["-ac", "1"]
        out_channels = 1
    else:
        out_channels = src_channels

    cmd += ["-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg decode failed for {p}: {result.stderr.decode().strip()}")

    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise AudioError(f"Decoded zero samples from {p} (window may be outside the file)")
    if out_channels > 1:
        samples = samples[: (samples.size // out_channels) * out_channels]
        samples = samples.reshape(-1, out_channels)
    return samples, sr


def extract_segment_to_file(
    path: str,
    start: float,
    end: float,
    output_path: str | None = None,
    overwrite: bool = False,
) -> str:
    """Extract [start, end] to a 16-bit wav file and return its path.

    Refuses to overwrite an existing file unless overwrite=True.
    """
    _require_ffmpeg()
    p = resolve_path(path)
    meta = audio_summary(path)
    start, end = validate_window(start, end, meta["duration_seconds"])
    if end is None:
        raise AudioError("end_time is required")
    if output_path:
        out = Path(output_path).expanduser().resolve()
    else:
        out = p.with_name(f"{p.stem}_{start:.1f}s-{end:.1f}s.wav")
    if out == p:
        raise AudioError("output_path must differ from the source file")
    if out.exists() and not overwrite:
        raise AudioError(f"{out} already exists; pass overwrite=true to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{start:.3f}", "-i", str(p), "-t", f"{end - start:.3f}",
            "-acodec", "pcm_s16le",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioError(f"ffmpeg extract failed: {result.stderr.strip()}")
    return str(out)
