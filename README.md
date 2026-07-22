# voice-analysis-mcp

An MCP server that gives LLMs like Claude the tools to review and do
**qualitative analysis of call recordings, voice memos, TTS output, and other
audio files** — transcription, conversation dynamics, vocal delivery, recording
quality, and visual inspection via spectrograms/waveforms.

Everything runs locally: ffmpeg for decoding, [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
for transcription (models download on first use), librosa for signal analysis.
No API keys required.

## Tools

| Tool | What it answers |
| --- | --- |
| `get_audio_info` | What is this file? Duration, channels, sample rate, codec, tags. |
| `transcribe` | What was said, when? Timestamped segments, optional word timing, per-channel (per-speaker) transcription for stereo calls, words-per-minute. |
| `analyze_conversation` | Who talks how much? Talk time and turns per channel, dead air, overlap, interruption counts, response latency (stereo calls: one speaker per channel). |
| `analyze_prosody` | How does it sound? Pitch median/range (monotone vs expressive), loudness and dynamic range, pace, pause/hesitation patterns for a window (≤ 300 s). |
| `analyze_quality` | Can I trust this recording? Levels, clipping, noise floor, estimated SNR, narrowband detection, flagged issues. |
| `extract_segment` | Cut a key moment out to a standalone wav file. |
| `render_spectrogram` | A mel spectrogram image the model can look at — spot hold music, DTMF/beeps, hum, dropouts, TTS artifacts. |
| `render_waveform` | Per-channel waveform image — speaker activity, silences, level imbalance at a glance. |

A typical call review: `get_audio_info` → `analyze_conversation` (whole file,
cheap) → `transcribe` (per channel for stereo) → `analyze_prosody` /
`render_spectrogram` on the interesting windows.

## Requirements

- Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/)
- ffmpeg on PATH (`brew install ffmpeg`)

## Setup

```bash
uv sync
uv run pytest        # optional: verify
```

### Claude Code

```bash
claude mcp add voice-analysis -- uv run --directory /path/to/voice-analysis.mcp voice-analysis-mcp
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "voice-analysis": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/voice-analysis.mcp", "voice-analysis-mcp"]
    }
  }
}
```

## Notes & design choices

- **Stereo = per-speaker.** Telephony platforms typically export calls with one
  speaker per channel. Tools take a `channel` parameter to isolate a speaker;
  `analyze_conversation` uses both channels for interruption/overlap/latency
  metrics. Mono recordings fall back to overall speech/silence dynamics —
  model-based diarization (e.g. pyannote) is a possible future addition.
- **Whisper models**: `tiny`/`base`/`small`/`medium`/`large-v3`/`large-v3-turbo`,
  cached under the Hugging Face cache dir after first download. `base` is the
  default trade-off; use `small`+ for noisy telephone audio.
- **Windows over whole files** for the expensive analyses: prosody is capped at
  300 s and images at 600 s per call — pass `start_time`/`end_time`. All
  timestamps in results are absolute seconds into the file, so findings from
  different tools line up.
- **Any input format** works if ffmpeg can decode it, including pulling the
  audio track out of video files.
