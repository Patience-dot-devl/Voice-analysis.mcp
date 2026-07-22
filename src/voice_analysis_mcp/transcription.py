"""Transcription via faster-whisper (runs locally, no API key needed).

Models download to the Hugging Face cache on first use:
tiny ~75 MB, base ~145 MB, small ~480 MB, medium ~1.5 GB, large-v3 ~3 GB.
"""

from __future__ import annotations

import tempfile

import numpy as np

from .audio_io import AudioError, load_audio

VALID_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
_models: dict[str, object] = {}


def _get_model(model_size: str):
    if model_size not in VALID_MODELS:
        raise AudioError(f"model_size must be one of {VALID_MODELS}, got {model_size!r}")
    if model_size not in _models:
        from faster_whisper import WhisperModel  # heavy import, keep local

        _models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _models[model_size]


def transcribe(
    path: str,
    model_size: str = "base",
    language: str | None = None,
    word_timestamps: bool = False,
    channel: int | None = None,
    start: float | None = None,
    end: float | None = None,
) -> dict:
    y, sr = load_audio(
        path, sample_rate=16000, mono=True, start=start, end=end, channel=channel
    )

    model = _get_model(model_size)
    offset = start or 0.0

    # faster-whisper accepts a float32 array directly at 16 kHz
    segments_iter, info = model.transcribe(
        np.ascontiguousarray(y),
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=True,
    )

    segments = []
    word_count = 0
    for seg in segments_iter:
        item = {
            "start": round(seg.start + offset, 2),
            "end": round(seg.end + offset, 2),
            "text": seg.text.strip(),
        }
        word_count += len(seg.text.split())
        if word_timestamps and seg.words:
            item["words"] = [
                {"start": round(w.start + offset, 2), "end": round(w.end + offset, 2), "word": w.word}
                for w in seg.words
            ]
        segments.append(item)

    speech_duration = sum(s["end"] - s["start"] for s in segments)
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "model": model_size,
        "channel": channel,
        "audio_duration_seconds": round(info.duration + offset, 2),
        "words_per_minute": round(word_count / (speech_duration / 60), 1) if speech_duration else None,
        "segment_count": len(segments),
        "segments": segments,
    }
