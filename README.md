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
| `transcribe` | What was said, when? Timestamped segments, optional word timing, words-per-minute. `split_channels=true` returns one merged who-said-what transcript for stereo calls; `channel` isolates one side. Output capped at `max_segments` with a `next_start_time` cursor. |
| `analyze_conversation` | Who talks how much? Talk time and turns per channel, dead air, overlap, interruptions vs. backchannels, response latency (stereo calls: one speaker per channel). |
| `analyze_prosody` | How does it sound? Pitch median/range (monotone vs expressive), loudness and dynamic range, estimated syllable rate, pause/hesitation patterns for a window (≤ 300 s). |
| `analyze_quality` | Can I trust this recording? Levels (overall and per channel), clipping, noise floor, estimated SNR, effective bandwidth (catches upsampled telephone audio), channel imbalance, flagged issues. |
| `extract_segment` | Cut a key moment out to a standalone wav file. Never overwrites unless `overwrite=true`. |
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
uv run pytest        # optional: verify (transcription tests skip unless the tiny Whisper model is cached)
uv run ruff check .  # lint
```

### Claude Code

Working inside this repo, nothing to configure: the checked-in [`.mcp.json`](.mcp.json)
registers the server automatically (approve it when prompted). From any other
project, register it globally:

```bash
claude mcp add --scope user voice-analysis -- uv run --directory /path/to/voice-analysis.mcp voice-analysis-mcp
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

## How a Claude instance uses this

The model only needs a **file path** — every tool takes an absolute path to a
local audio file, so "analyze the call in ~/Downloads/call-4711.mp3" is enough
to start. The server's instructions steer the model through the intended flow:

1. `get_audio_info` — learn duration and channel count (drives everything else)
2. `analyze_conversation` — cheap whole-file pass: talk balance, dead air, interruptions
3. `transcribe` — content, per channel on stereo calls for who-said-what
4. `analyze_prosody` / `analyze_quality` / `render_spectrogram` on windows the
   first passes flagged — all results use absolute seconds, so findings from
   different tools line up

Example prompts that work end-to-end:

> Review ~/calls/support-0412.wav: who dominated the conversation, were there
> awkward silences, and how did the agent's tone change after the customer
> pushed back?

> Compare the TTS output in v1.wav and v2.wav — which sounds less monotone and
> are there any audible artifacts?

Results are compact JSON (rounded numbers, capped lists) sized for a model's
context window; images come back as MCP image content Claude can view directly.

## Notes & design choices

- **Stereo = per-speaker.** Telephony platforms typically export calls with one
  speaker per channel. Tools take a `channel` parameter to isolate a speaker;
  `analyze_conversation` uses both channels for interruption/overlap/latency
  metrics. Mono recordings fall back to overall speech/silence dynamics —
  model-based diarization (e.g. pyannote) is a possible future addition.
- **Whisper models**: `tiny`/`base`/`small`/`medium`/`large-v3`/`large-v3-turbo`,
  cached under the Hugging Face cache dir after first download. `base` is the
  default trade-off; use `small`+ for noisy telephone audio. Inference runs on
  CPU with int8 weights by default; set `VOICE_ANALYSIS_WHISPER_DEVICE=cuda`
  and `VOICE_ANALYSIS_WHISPER_COMPUTE_TYPE=float16` for a GPU.
- **Long transcripts** are capped at `max_segments` (default 400). The
  response then carries `truncated: true` and `next_start_time`; call again
  from there.
- **Non-blocking tools.** Every tool runs its CPU work in a worker thread, so
  the server keeps answering pings and cancellations during a long transcription.
- **Windows over whole files** for the expensive analyses: prosody is capped at
  300 s and images at 600 s per call — pass `start_time`/`end_time`. All
  timestamps in results are absolute seconds into the file, so findings from
  different tools line up. Windows outside the file are rejected rather than
  silently clamped.
- **Energy-based VAD** drives the conversation, prosody and quality metrics. It
  adapts to the recording's noise floor, treats overlaps under 0.3 s as
  backchannels rather than interruptions, and only counts a response latency
  when the first speaker really finished (no resume before the reply).
- **Any input format** works if ffmpeg can decode it, including pulling the
  audio track out of video files.
