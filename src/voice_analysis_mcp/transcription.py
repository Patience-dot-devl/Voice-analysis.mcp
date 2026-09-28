"""Transcription via faster-whisper (runs locally, no API key needed).

Models download to the Hugging Face cache on first use:
tiny ~75 MB, base ~145 MB, small ~480 MB, medium ~1.5 GB, large-v3 ~3 GB.

Environment overrides for the inference backend:
  VOICE_ANALYSIS_WHISPER_DEVICE        cpu (default) | cuda | auto
  VOICE_ANALYSIS_WHISPER_COMPUTE_TYPE  int8 (default) | float16 | int8_float16 | ...
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .audio_io import AudioError, audio_summary, load_audio

VALID_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
DEFAULT_MAX_SEGMENTS = 400
_models: dict[str, object] = {}
_models_lock = threading.Lock()


def _get_model(model_size: str):
    if model_size not in VALID_MODELS:
        raise AudioError(f"model_size must be one of {VALID_MODELS}, got {model_size!r}")
    with _models_lock:
        if model_size not in _models:
            from faster_whisper import WhisperModel  # heavy import, keep local

            _models[model_size] = WhisperModel(
                model_size,
                device=os.environ.get("VOICE_ANALYSIS_WHISPER_DEVICE", "cpu"),
                compute_type=os.environ.get("VOICE_ANALYSIS_WHISPER_COMPUTE_TYPE", "int8"),
            )
        return _models[model_size]


def transcribe(
    path: str,
    model_size: str = "base",
    language: str | None = None,
    word_timestamps: bool = False,
    channel: int | None = None,
    split_channels: bool = False,
    start: float | None = None,
    end: float | None = None,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> dict:
    """Transcribe a file or window.

    split_channels=True transcribes every channel separately and merges the
    segments by start time with a `speaker` label ("channel_0", ...), which is
    the who-said-what view for stereo call recordings.
    """
    if split_channels and channel is not None:
        raise AudioError("pass either channel or split_channels, not both")
    if max_segments < 1:
        raise AudioError("max_segments must be >= 1")

    model = _get_model(model_size)
    offset = start or 0.0

    if split_channels:
        n_channels = audio_summary(path)["channels"]
        if n_channels < 2:
            raise AudioError("split_channels requires a multi-channel file; this one is mono")
        channels = list(range(n_channels))
    else:
        channels = [channel]

    segments: list[dict] = []
    word_count = 0
    infos = []
    for ch in channels:
        y, _sr = load_audio(path, sample_rate=16000, mono=True, start=start, end=end, channel=ch)
        # faster-whisper accepts a float32 array directly at 16 kHz
        segments_iter, info = model.transcribe(
            np.ascontiguousarray(y),
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )
        infos.append(info)
        for seg in segments_iter:
            item = {
                "start": round(seg.start + offset, 2),
                "end": round(seg.end + offset, 2),
                "text": seg.text.strip(),
            }
            if split_channels:
                item["speaker"] = f"channel_{ch}"
            word_count += len(seg.text.split())
            if word_timestamps and seg.words:
                item["words"] = [
                    {"start": round(w.start + offset, 2), "end": round(w.end + offset, 2), "word": w.word}
                    for w in seg.words
                ]
            segments.append(item)

    if split_channels:
        segments.sort(key=lambda s: s["start"])

    truncated = len(segments) > max_segments
    if truncated:
        segments = segments[:max_segments]

    speech_duration = sum(s["end"] - s["start"] for s in segments)
    words_in_output = sum(len(s["text"].split()) for s in segments)
    info = infos[0]
    result = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "model": model_size,
        "channel": channel,
        "split_channels": split_channels,
        "audio_duration_seconds": round(info.duration + offset, 2),
        "words_per_minute": round(words_in_output / (speech_duration / 60), 1) if speech_duration else None,
        "segment_count": len(segments),
        "segments": segments,
    }
    if truncated:
        result["truncated"] = True
        result["next_start_time"] = segments[-1]["end"]
        result["note"] = (
            f"Output capped at {max_segments} segments. Call again with "
            f"start_time={segments[-1]['end']} for the rest, or raise max_segments."
        )
    return result
