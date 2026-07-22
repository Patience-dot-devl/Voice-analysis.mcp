"""Render spectrogram / waveform PNGs so the LLM can inspect audio visually."""

from __future__ import annotations

import io

import numpy as np

from .audio_io import AudioError, load_audio

MAX_WINDOW_S = 600.0
ANALYSIS_SR = 16000


def _check_window(y: np.ndarray, sr: int) -> float:
    duration = y.size / sr
    if duration > MAX_WINDOW_S:
        raise AudioError(
            f"Render window is {duration:.0f}s; max is {MAX_WINDOW_S:.0f}s. "
            "Pass start_time/end_time to render a shorter window."
        )
    return duration


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def spectrogram_png(
    path: str,
    start: float | None = None,
    end: float | None = None,
    channel: int | None = None,
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    y, sr = load_audio(path, sample_rate=ANALYSIS_SR, mono=True, start=start, end=end, channel=channel)
    _check_window(y, sr)
    offset = start or 0.0

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=sr // 2, ax=ax, cmap="magma"
    )
    xlim = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t + offset:.1f}".rstrip("0").rstrip(".") for t in ticks])
    ax.set_xlabel("time (s, absolute)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel spectrogram")
    return _fig_to_png(fig)


def waveform_png(
    path: str,
    start: float | None = None,
    end: float | None = None,
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr = load_audio(path, sample_rate=ANALYSIS_SR, mono=False, start=start, end=end)
    if y.ndim == 1:
        y = y[:, None]
    _check_window(y[:, 0], sr)
    offset = start or 0.0
    n_channels = y.shape[1]

    fig, axes = plt.subplots(
        n_channels, 1, figsize=(12, 2.2 * n_channels), sharex=True, squeeze=False
    )
    t = offset + np.arange(y.shape[0]) / sr
    step = max(1, y.shape[0] // 12000)  # decimate for plotting
    for ch in range(n_channels):
        ax = axes[ch][0]
        ax.plot(t[::step], y[::step, ch], linewidth=0.4)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(True, alpha=0.3)
    axes[-1][0].set_xlabel("time (s, absolute)")
    axes[0][0].set_title("Waveform" + (" (one speaker per channel?)" if n_channels == 2 else ""))
    return _fig_to_png(fig)
