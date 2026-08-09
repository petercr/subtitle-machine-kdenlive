# subtitle-machine-kdenlive
An app/MCP that allows for programmatic use of Kdenlive by MCP to add subtitles to videos.

## Development setup

The project targets Python 3.11+ and uses uv to manage its local virtual
environment and locked dependencies. Development currently pins Python 3.12.

```powershell
uv sync
uv run video-mcp --version
uv run pytest
```

Copy `video-mcp.example.yaml` to the machine-local `video-mcp.yaml` when you
want to customize executable, model, or output paths. The local file is ignored
by Git. Environment variables such as `VIDEO_MCP_FFMPEG`,
`VIDEO_MCP_WHISPER_CPP`, `VIDEO_MCP_ASR_MODEL`, `VIDEO_MCP_ASR_DEVICE`, and
`VIDEO_MCP_WORKSPACE` override YAML values.

Print the effective configuration with:

```powershell
uv run video-mcp --config video-mcp.example.yaml config
```

Run the local environment diagnostic with:

```powershell
uv run video-mcp --config video-mcp.example.yaml doctor
uv run video-mcp --config video-mcp.example.yaml doctor --json
```

Inspect a source video and extract the normalized ASR audio with:

```powershell
uv run video-mcp --config video-mcp.example.yaml inspect "C:\Videos\Test Video.mp4"
uv run video-mcp --config video-mcp.example.yaml extract-audio "C:\Videos\Test Video.mp4"
```

The inspection output is normalized application data from FFprobe. Audio
extraction creates a mono, 16 kHz, 16-bit PCM WAV in the configured workspace;
existing output is preserved unless `--overwrite` is supplied.

Transcribe a normalized audio file with the configured Whisper.cpp model:

```powershell
uv run video-mcp --config video-mcp.example.yaml transcribe "work\Test Video.wav" --device cpu
```

This writes a versioned `*.transcript.raw.json` file containing segment and
token timestamps. The source audio is never modified.

Export normalized transcript data as SRT:

```powershell
uv run video-mcp --config video-mcp.example.yaml export-srt "work\Test Video.transcript.raw.json"
```

SRT cue numbers are generated deterministically, timestamps are validated for
ordering and overlap, and existing output is preserved unless `--overwrite` is
supplied.

Export the same transcript as styled ASS for FFmpeg rendering or Kdenlive:

```powershell
uv run video-mcp --config video-mcp.example.yaml export-ass "work\Test Video.transcript.raw.json"
```

The initial `clean` preset uses a readable white Arial style with outline and
shadow settings, and the ASS play resolution can be adjusted with `--width`
and `--height`.

Create a fast burned-in preview from the source video and ASS file:

```powershell
uv run video-mcp --config video-mcp.example.yaml create-preview `
  "C:\Videos\Test Video.mp4" `
  "work\Test Video.ass"
```

Preview rendering uses FFmpeg, preserves the source video, and defaults to a
1280-pixel-wide H.264/AAC output. Use `--width` and `--overwrite` as needed.

Run the complete local caption pipeline with one command:

```powershell
uv run video-mcp --config video-mcp.example.yaml caption `
  "C:\Videos\Test Video.mp4" `
  --device cpu
```

This creates a job directory containing `source.json`, normalized audio,
`transcript.raw.json`, `transcript.cleaned.json`, `subtitles.srt`,
`subtitles.ass`, and `captioned-preview.mp4`. Re-running reuses existing
artifacts; use `--overwrite` to regenerate them.

Create an editable Kdenlive project from the generated SRT:

```powershell
uv run video-mcp --config video-mcp.example.yaml kdenlive `
  "C:\Videos\Test Video.mp4" `
  --subtitles "work\Test Video\subtitles.srt"
```

The project defaults to `work\Test Video-captioned.kdenlive` and writes the
required sibling `Test Video-captioned.kdenlive.srt`. Use `--output` to choose
another project path and `--overwrite` to replace existing project assets.
Editable Kdenlive export currently requires SRT; ASS remains the asset used
for FFmpeg burned-in previews.

# Codex Implementation Brief — Windows Local Video Subtitle MCP

## Goal

Build a local-first Windows application/MCP server that can:

1. Take an existing video file.
2. Transcribe its speech locally.
3. Generate well-formatted subtitles.
4. Produce SRT and ASS subtitle files.
5. Render a subtitled preview/final video using FFmpeg.
6. Optionally create an editable Kdenlive project containing the video and subtitles.
7. Expose these operations as deterministic MCP tools for Codex/Claude/other MCP clients.

The application must work without cloud APIs.

Primary target environment:

* Windows 10/11 x64
* Kdenlive installed locally
* CPU-first execution
* NVIDIA GTX 1660-class GPU with ~4 GB VRAM available as optional acceleration
* Local filesystem input/output

Do not design around requiring the GPU.

---

# Architectural Principle

Do NOT build a remote-control wrapper around the Kdenlive GUI.

Kdenlive is an optional editor/handoff target.

The core product is a deterministic local video-processing pipeline:

```text
Video
  ↓
Media inspection
  ↓
Audio extraction
  ↓
Local ASR
  ↓
Normalized subtitle data
  ↓
Subtitle cleanup / formatting
  ↓
SRT + ASS
  ↓
 ┌───────────────┬─────────────────┐
 ↓               ↓
FFmpeg render    Kdenlive project
 ↓               ↓
MP4             Editable project
```

MCP sits on top of this pipeline.

Every core operation must also be callable directly from Python without MCP.

---

# Technology Stack

Use Python 3.11+.

Use the current official MCP Python SDK v2.

Initial MCP transport:

```text
stdio
```

Do not make HTTP transport necessary for v1.

Core external executables:

```text
ffmpeg.exe
ffprobe.exe
whisper.cpp executable
Kdenlive / MLT tooling where appropriate
```

Prefer subprocess invocation of stable native tools over large Python ML dependency stacks.

Use pathlib for all filesystem operations.

Windows paths containing spaces must work correctly.

Never construct shell command strings. Pass subprocess arguments as arrays.

---

# ASR Backends

Define an abstraction:

```python
class ASRBackend:
    def transcribe(audio_path, options) -> Transcript:
        ...
```

Implement at least:

```text
WhisperCppBackend
```

Design for later:

```text
ParakeetBackend
```

## Whisper.cpp

Whisper.cpp is the v1 baseline.

It must support:

```text
CPU
CUDA if available
```

GPU failure must never make transcription impossible.

Desired behavior:

```text
auto
 ↓
try CUDA backend if configured
 ↓
if unavailable/fails
 ↓
CPU
```

Do not auto-download large models without an explicit command/tool.

Model locations should be configurable.

Initial useful Whisper models:

```text
base
small
```

Benchmark later to determine default.

## Parakeet

Treat OpenASR/Parakeet 0.6B as an experimental backend.

Do not make it a dependency of milestone 1.

Investigate whether a clean native Windows deployment exists.

If it works reliably:

```text
Parakeet Q8 CPU
```

may become the preferred fast transcription backend.

The backend interface must make switching ASR engines trivial.

---

# Internal Subtitle Data Model

Do NOT make SRT the application's internal source of truth.

Define normalized structures similar to:

```python
@dataclass
class Word:
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None

@dataclass
class SubtitleSegment:
    id: str
    start_ms: int
    end_ms: int
    text: str
    words: list[Word]
    speaker: str | None = None

@dataclass
class Transcript:
    language: str | None
    duration_ms: int
    segments: list[SubtitleSegment]
```

Persist normalized transcripts as JSON.

Example:

```text
work/
  video-name/
    source.json
    transcript.raw.json
    transcript.cleaned.json
    subtitles.srt
    subtitles.ass
```

This JSON representation should be stable and versioned.

---

# Caption Formatting Engine

Build deterministic caption segmentation before involving any LLM.

Config example:

```yaml
max_chars_per_line: 42
max_lines: 2
min_duration_ms: 700
max_duration_ms: 6000
min_gap_ms: 80
prefer_sentence_boundaries: true
prefer_phrase_boundaries: true
```

The formatter should try to avoid:

* one-word dangling lines
* splitting proper names
* splitting immediately before punctuation
* excessively rapid captions
* overlapping subtitle timestamps

Keep word timestamps whenever ASR supplies them.

The formatter should be thoroughly unit tested.

---

# Subtitle Styling

Support named presets.

Example:

```text
clean
shorts-bold
interview
agency-default
```

Represent styles as configuration rather than generated commands.

Example conceptual structure:

```yaml
name: shorts-bold
font: Arial
font_size: 64
alignment: bottom-center
margin_bottom: 120
outline: 4
shadow: 1
max_lines: 2
```

Generate ASS from these presets.

Do not hardcode styling into FFmpeg command construction.

---

# Local LLM Cleanup

LLM cleanup is optional.

The pipeline must work without an LLM.

Define:

```text
SubtitleCleaner
```

implementations:

```text
DeterministicCleaner
LocalLLMCleaner
```

Eventually use a small GGUF model through llama.cpp.

Candidate:

```text
Qwen3.5 2B Q4
```

But do not make this required for milestone 1.

LLM responsibilities should be narrowly constrained:

* punctuation correction
* capitalization
* obvious ASR error correction
* sentence boundary recovery

The LLM must NOT:

* summarize
* paraphrase
* invent dialogue
* alter meaning

Require structured JSON output and validate it before accepting changes.

Original transcription must always be preserved.

---

# FFmpeg Adapter

Implement:

```text
probe_video()
extract_audio()
render_subtitles()
create_preview()
```

Audio extraction target:

```text
mono
16 kHz
PCM WAV
```

FFprobe result should capture at least:

```text
duration
width
height
frame rate
video codec
audio codec
sample rate
rotation/orientation
```

Rendering must support:

```text
soft subtitle output
burned-in subtitle output
```

For preview generation, allow reduced resolution / faster encoding.

Never overwrite the original video.

---

# Kdenlive Adapter

Kdenlive support is deliberately downstream from the core pipeline.

Initial Kdenlive goal:

```text
source video
+
generated subtitle track
+
correct project settings
=
editable .kdenlive project
```

Do not automate the Kdenlive GUI using mouse/keyboard controls unless absolutely unavoidable.

Prefer:

1. supported project/MLT structures
2. MLT tooling
3. existing open-source Kdenlive automation code
4. GUI automation only as a last resort

Current development should target Kdenlive 26.x on Windows.

Research existing open-source:

```text
Kdenlive MCP servers
Kdenlive CLI wrappers
MLT project generators
CLI-Anything Kdenlive implementation
```

Before copying code:

* inspect license
* document provenance
* identify reusable components
* avoid importing an entire architecture unnecessarily

Create an adapter boundary:

```python
class ProjectAdapter:
    def create_project(...): ...
    def add_video(...): ...
    def add_subtitles(...): ...
    def save(...): ...
```

Implementation:

```text
KdenliveProjectAdapter
```

The rest of the application must not depend directly on Kdenlive XML internals.

---

# MCP Server

Keep MCP thin.

MCP tools should call application services.

Do not put media-processing logic inside MCP handlers.

Initial tool surface:

```text
video.inspect

video.transcribe

video.generate_subtitles

video.create_preview

video.render

subtitle.export_srt

subtitle.export_ass

project.create_kdenlive
```

Possible convenience tool:

```text
video.caption
```

which orchestrates:

```text
inspect
→ transcribe
→ format
→ export
→ optionally render
```

Every long-running tool should return useful structured status/output information.

Example response:

```json
{
  "success": true,
  "input": "C:\\Videos\\demo.mp4",
  "transcript": "C:\\Videos\\demo.work\\transcript.cleaned.json",
  "srt": "C:\\Videos\\demo.work\\subtitles.srt",
  "ass": "C:\\Videos\\demo.work\\subtitles.ass",
  "rendered_video": null,
  "warnings": []
}
```

---

# Proposed Repository Structure

```text
video-subtitle-mcp/
│
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
│
├─ src/
│  └─ video_mcp/
│     │
│     ├─ config.py
│     ├─ models.py
│     │
│     ├─ media/
│     │  ├─ ffmpeg.py
│     │  └─ probe.py
│     │
│     ├─ asr/
│     │  ├─ base.py
│     │  ├─ whisper_cpp.py
│     │  └─ parakeet.py
│     │
│     ├─ subtitles/
│     │  ├─ formatter.py
│     │  ├─ srt.py
│     │  ├─ ass.py
│     │  ├─ styles.py
│     │  └─ cleaner.py
│     │
│     ├─ adapters/
│     │  ├─ ffmpeg.py
│     │  └─ kdenlive.py
│     │
│     ├─ services/
│     │  ├─ transcription.py
│     │  ├─ captioning.py
│     │  └─ rendering.py
│     │
│     └─ mcp/
│        └─ server.py
│
├─ presets/
│  ├─ clean.yaml
│  └─ shorts-bold.yaml
│
├─ tests/
│
└─ fixtures/
```

---

# Configuration

Support a project-level config file such as:

```text
video-mcp.yaml
```

Example:

```yaml
tools:
  ffmpeg: "C:/Tools/ffmpeg/bin/ffmpeg.exe"
  ffprobe: "C:/Tools/ffmpeg/bin/ffprobe.exe"
  whisper_cpp: "C:/Tools/whisper/whisper-cli.exe"
  kdenlive: "C:/Program Files/kdenlive/bin/kdenlive.exe"

asr:
  backend: whisper_cpp
  device: auto
  model: "C:/Models/whisper/ggml-small.bin"

subtitles:
  preset: clean
  max_chars_per_line: 42
  max_lines: 2

output:
  workspace: "./work"
```

Also support environment-variable overrides.

Do not make users edit Python source to configure executable/model paths.

---

# Hardware Detection

Implement a diagnostic command/service:

```text
video-mcp doctor
```

It should report:

```text
Windows version
CPU
system RAM

FFmpeg found?
FFprobe found?

whisper.cpp found?
Whisper model found?

CUDA/NVIDIA GPU detectable?
GPU name
VRAM if detectable

Kdenlive found?
Kdenlive version

MLT/melt available?

llama.cpp available?

workspace writable?
```

Do not fail because optional components are absent.

Report capabilities.

Example:

```text
Core caption pipeline: READY
Whisper CPU: READY
Whisper CUDA: READY
Parakeet: NOT INSTALLED
Kdenlive export: READY
Local LLM cleanup: NOT INSTALLED
```

---

# Milestones

## Milestone 0 — Research and Spike

Before committing architecture around existing Kdenlive projects:

1. Identify the strongest existing open-source Kdenlive MCP/CLI projects.
2. Inspect their licenses.
3. Determine how they manipulate Kdenlive/MLT projects.
4. Test creating a minimal Kdenlive project programmatically on Windows.
5. Document what should be reused versus rewritten.

Output:

```text
docs/kdenlive-research.md
```

Do not spend excessive time making the existing MCP server work if its architecture is unsuitable.

---

## Milestone 1 — Core Caption Pipeline

Must work completely without MCP or Kdenlive.

CLI:

```text
video-mcp caption input.mp4
```

produces:

```text
transcript.raw.json
transcript.cleaned.json
subtitles.srt
subtitles.ass
captioned-preview.mp4
```

Use:

```text
FFmpeg
whisper.cpp
deterministic formatter
```

No local LLM yet.

Acceptance criteria:

* Windows paths with spaces work.
* Original video remains untouched.
* CPU transcription works.
* Generated captions contain valid timestamps.
* No overlapping captions.
* SRT opens correctly in common players.
* ASS burns successfully through FFmpeg.
* Interrupted jobs produce understandable errors.
* Re-running is safe/idempotent where reasonable.

---

## Milestone 2 — GPU Acceleration

Enable whisper.cpp CUDA.

Device selection:

```text
cpu
cuda
auto
```

Acceptance criteria:

* GTX 1660 can be detected where supported.
* `auto` falls back to CPU cleanly.
* GPU failure never corrupts job output.
* Benchmark CPU versus CUDA.

Create:

```text
benchmarks/asr-results.md
```

Measure:

```text
wall-clock transcription time
real-time factor
peak RAM
GPU VRAM
```

---

## Milestone 3 — Kdenlive Project Export

Command:

```text
video-mcp kdenlive input.mp4 --subtitles subtitles.srt
```

should produce:

```text
input-captioned.kdenlive
```

Opening that project manually in Kdenlive should show:

* original media
* correct resolution/frame rate
* synchronized captions/subtitle track
* editable caption content
* no missing-media errors

No GUI automation should be needed to create it.

---

## Milestone 4 — MCP Interface

Wrap the tested service layer with MCP.

Expose:

```text
video.inspect
video.transcribe
video.generate_subtitles
video.create_preview
video.render
project.create_kdenlive
```

Test through an MCP client.

MCP failure must not leave orphaned FFmpeg/ASR processes.

---

## Milestone 5 — Parakeet Evaluation

Test OpenASR Parakeet 0.6B Q8 on Windows.

Compare against Whisper on several representative videos.

Measure:

```text
transcription speed
RAM
accuracy
word timestamp quality
punctuation
installation complexity
Windows reliability
```

Only promote Parakeet to default if the complete Windows experience is clearly better.

---

## Milestone 6 — Local LLM Cleanup

Add llama.cpp adapter.

Evaluate a small Qwen model.

Give the model small transcript chunks rather than entire videos.

Validate every response against strict JSON schemas.

Keep deterministic cleanup as fallback.

---

# Testing Strategy

Use pytest.

Unit-test heavily:

```text
subtitle segmentation
timestamp conversion
line wrapping
ASS escaping
SRT generation
configuration
path handling
model serialization
```

Integration tests should cover:

```text
FFprobe
FFmpeg audio extraction
Whisper invocation
FFmpeg subtitle rendering
Kdenlive project generation
```

Include one very small media fixture suitable for repository tests if licensing permits.

Do not require downloading gigabytes of models in CI.

Use mock ASR output for normal CI tests.

---

# Logging

Use structured logging.

Each processing job should get a job ID.

Log:

```text
input file
detected media metadata
backend used
model used
CPU/GPU device
processing duration
generated outputs
warnings/errors
```

Never silently switch transcription models.

If `auto` falls back from CUDA to CPU, explicitly report that.

---

# Error Handling

Create typed application errors such as:

```text
ExecutableNotFound
ModelNotFound
UnsupportedMedia
TranscriptionFailed
SubtitleGenerationFailed
RenderFailed
KdenliveProjectFailed
```

MCP handlers should convert these into useful user-facing messages.

Avoid dumping giant subprocess traces unless debug mode is enabled.

---

# Things NOT to Build Yet

Do not initially build:

* Kdenlive GUI automation
* live OBS captioning
* speaker diarization
* word-by-word TikTok animation
* cloud transcription
* cloud LLM integration
* video cutting/editing
* automatic B-roll
* automatic scene detection
* web frontend
* database
* authentication
* distributed jobs

Keep v1 extremely focused.

---

# Definition of Initial Success

The first meaningful demo should be:

```text
video-mcp caption "C:\Videos\Test Video.mp4"
```

and approximately one command later we have:

```text
Test Video.work/
    transcript.raw.json
    transcript.cleaned.json
    subtitles.srt
    subtitles.ass
    preview.mp4
```

Then:

```text
video-mcp kdenlive "C:\Videos\Test Video.mp4"
```

produces an editable Kdenlive project using those same subtitle assets.

Finally an MCP client should be able to request:

"Caption this video using the clean preset and make me an editable Kdenlive project."

The MCP server should perform the same deterministic pipeline without duplicating implementation logic.

---

# First Task for Codex

Begin with Milestone 0 and Milestone 1.

Before implementing:

1. Inspect the existing repository if one exists.
2. Check for existing Kdenlive MCP/MLT code worth reusing.
3. Verify licenses before copying code.
4. Create the proposed module boundaries.
5. Implement `video-mcp doctor`.
6. Implement FFprobe inspection.
7. Implement FFmpeg audio extraction.
8. Implement the Whisper.cpp adapter.
9. Define normalized transcript models.
10. Implement deterministic SRT generation.
11. Implement ASS generation with one `clean` preset.
12. Render a captioned preview through FFmpeg.
13. Add tests.
14. Document exact Windows setup in README.

Do not implement the MCP layer until the direct Python/CLI pipeline is working and tested.

Make reasonable implementation decisions autonomously. Keep dependencies minimal. Prefer boring, inspectable code over clever abstractions.
