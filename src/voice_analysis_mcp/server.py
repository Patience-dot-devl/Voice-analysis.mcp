"""MCP server exposing audio/call analysis tools over stdio."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP, Image

from . import conversation, prosody, quality, transcription, visuals
from .audio_io import audio_summary, extract_segment_to_file

mcp = FastMCP(
    "voice-analysis",
    instructions=(
        "Tools for qualitative analysis of call recordings, voice memos, TTS output, "
        "and other audio files. Typical workflow: get_audio_info first (duration, "
        "channels), then transcribe for content, analyze_conversation for talk-time/"
        "interruption dynamics, analyze_prosody on interesting windows for delivery "
        "(pitch, pace, pausing), analyze_quality for recording problems, and "
        "render_spectrogram/render_waveform to inspect the audio visually. "
        "Stereo call recordings usually have one speaker per channel — use the "
        "channel parameter to analyze each speaker separately. All times are seconds "
        "from the start of the file."
    ),
)


@mcp.tool()
def get_audio_info(path: str) -> dict:
    """Get metadata for an audio/video file: duration, channels, sample rate, codec, tags.

    Call this first for any new file — duration tells you how to window later
    analyses, and channels tells you whether per-speaker (stereo) analysis is possible.
    """
    return audio_summary(path)


@mcp.tool()
def transcribe(
    path: str,
    model_size: str = "base",
    language: str | None = None,
    word_timestamps: bool = False,
    channel: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict:
    """Transcribe speech to timestamped text using a local Whisper model (no API key).

    Returns segments with start/end times plus detected language and overall
    words-per-minute. On stereo call recordings, pass channel=0 or channel=1 to
    transcribe one speaker at a time and interleave the results by timestamp for
    an accurate who-said-what transcript. model_size: tiny/base/small/medium/
    large-v3/large-v3-turbo — larger is more accurate but slower; the model is
    downloaded on first use. word_timestamps=True adds per-word timing (useful
    for locating exact moments, at some cost in output size).
    """
    return transcription.transcribe(
        path,
        model_size=model_size,
        language=language,
        word_timestamps=word_timestamps,
        channel=channel,
        start=start_time,
        end=end_time,
    )


@mcp.tool()
def analyze_conversation(
    path: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict:
    """Measure conversation dynamics: talk time per channel, turns, dead air, overlap.

    On stereo call recordings (one speaker per channel) this also reports
    interruption counts per side, overlap ratio, and response latency —
    who dominates, who interrupts, how quickly each side responds. On mono
    audio only overall speech/silence dynamics are available. Cheap to run
    on a whole recording; a good early step for call review.
    """
    return conversation.analyze(path, start=start_time, end=end_time)


@mcp.tool()
def analyze_prosody(
    path: str,
    start_time: float | None = None,
    end_time: float | None = None,
    channel: int | None = None,
) -> dict:
    """Measure vocal delivery in a window (max 300s): pitch, energy, pace, pausing.

    Returns pitch statistics (median, range in semitones — low range means
    monotone delivery), loudness/dynamic range, speech ratio, and pause
    patterns (hesitations). Use on specific moments found via transcription or
    conversation analysis — e.g. compare an agent's delivery early vs late in a
    call, or check whether TTS output sounds flat. Pass channel to isolate one
    speaker on stereo recordings.
    """
    return prosody.analyze(path, start=start_time, end=end_time, channel=channel)


@mcp.tool()
def analyze_quality(
    path: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict:
    """Assess recording quality: levels, clipping, noise floor, estimated SNR.

    Flags common problems (clipping, very low level, high background noise,
    narrowband telephone audio). Run this before drawing conclusions from other
    analyses — poor quality audio degrades transcription and pitch tracking.
    """
    return quality.analyze(path, start=start_time, end=end_time)


@mcp.tool()
def extract_segment(
    path: str,
    start_time: float,
    end_time: float,
    output_path: str | None = None,
) -> dict:
    """Cut a time range out of a recording into a standalone wav file.

    Useful for isolating a key moment (an objection, an escalation, a TTS
    artifact) to share or analyze further. Returns the path of the new file;
    if output_path is omitted the file is written next to the source.
    """
    out = extract_segment_to_file(path, start_time, end_time, output_path)
    return {"output_path": out, "start_time": start_time, "end_time": end_time}


@mcp.tool()
def render_spectrogram(
    path: str,
    start_time: float | None = None,
    end_time: float | None = None,
    channel: int | None = None,
) -> Image:
    """Render a mel spectrogram image of a window (max 600s) for visual inspection.

    Look at this to spot things metrics miss: hold music, DTMF/beep tones,
    hum, dropouts, TTS glitches, where energy concentrates. Time axis is in
    absolute seconds so findings map back to transcript timestamps.
    """
    png = visuals.spectrogram_png(path, start=start_time, end=end_time, channel=channel)
    return Image(data=png, format="png")


@mcp.tool()
def render_waveform(
    path: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Image:
    """Render a waveform image (max 600s), one row per channel.

    On stereo call recordings the two rows show each speaker's activity at a
    glance — useful for spotting long silences, who talks when, clipping, and
    level imbalance between sides.
    """
    png = visuals.waveform_png(path, start=start_time, end=end_time)
    return Image(data=png, format="png")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
