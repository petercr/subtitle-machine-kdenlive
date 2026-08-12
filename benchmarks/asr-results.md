# ASR evaluation results

This file records the Windows comparison between the Whisper.cpp baseline and
the experimental Parakeet Q8 backend. Do not commit model weights or generated
audio here.

## Harness

The benchmark consumes normalized 16 kHz mono WAV files and emits one JSON
record per input:

```powershell
uv run python scripts/benchmark_asr.py `
  --config video-mcp.yaml `
  --backend whisper_cpp `
  --model "C:\Models\whisper\ggml-small.bin" `
  --reference-dir benchmarks/references `
  work\sample\audio.wav

uv run python scripts/benchmark_asr.py `
  --config video-mcp.yaml `
  --backend parakeet `
  --model "C:\Models\parakeet\ggml-parakeet-tdt-0.6b-v3-q8_0.bin" `
  --reference-dir benchmarks/references `
  work\sample\audio.wav
```

Place an optional `sample.txt` reference beside the matching `sample.wav`
under `benchmarks/references`. WER is case- and punctuation-insensitive.
Real-time factor is elapsed processing time divided by audio duration; lower
is faster. Results also include best-effort peak process RAM and GPU VRAM in
MiB. RAM is sampled from the ASR child process. VRAM is sampled from its PID
through `nvidia-smi` when available; Windows WDDM drivers that hide per-process
usage instead report a clearly labeled whole-device memory delta.

## Machine

| Date | Windows | CPU | RAM | GPU | Driver | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-12 | Windows 11 10.0.26200 | Intel64 Family 6 Model 158, 6 logical processors | 23.9 GB | NVIDIA GeForce GTX 1650 SUPER, 4096 MiB | 596.36 | Official whisper.cpp v1.8.5 cuBLAS 12.4 bundle; CUDA Toolkit/nvcc not installed |

## Results

| Backend | Model | Device | Audio | Elapsed (s) | RTF | Peak RAM (MiB) | Peak VRAM (MiB) | WER | Segments | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Whisper.cpp | ggml-small.bin | cpu | videorc-session.wav (19.200 s) | 17.546 | 0.914 | 935.4 | — | — | 4 | CUDA build with `--no-gpu`; warmed local run |
| Whisper.cpp | ggml-small.bin | cuda | videorc-session.wav (19.200 s) | 30.211 | 1.573 | 599.6 | 873.0 | — | 4 | CUDA device 0 active; VRAM is a `device_delta` because WDDM did not expose the ASR PID |
| Parakeet | tdt-0.6b-v3-q8_0 | cpu | videorc-session.wav (19.200 s) | 3.699 | 0.193 | 781.1 | — | — | 1 | whisper.cpp v1.9.2 CPU build; word timestamps formatted into 5 SRT cues |

Record qualitative timestamp and punctuation observations in the notes column
or below each run. Resource metrics are best-effort rather than a claim of
whole-system usage: RAM covers only the launched ASR process, and the JSON
`gpu_memory_scope` field distinguishes process VRAM from a WDDM device delta.

## Installation notes

The selected runtime is the native [`parakeet-cli`](https://github.com/ggml-org/whisper.cpp/tree/master/examples/parakeet-cli)
target from whisper.cpp, using the Q8_0 model published in
[`ggml-org/parakeet-GGUF`](https://huggingface.co/ggml-org/parakeet-GGUF).
See the upstream references before downloading or building the optional
runtime.

Current local status: Parakeet CPU runtime and Q8 model installed and validated
on Windows. The sample text broadly matches Whisper, with one differing proper
noun ("video orc" versus Whisper's "video work"); no reference transcript is
available for WER. Its raw output is one long segment, so the deterministic
word-timestamp formatter now splits it into five readable subtitle cues.

The current CUDA build initializes the GTX 1650 SUPER successfully. The
recorded clip is short and uses the small model, so its CUDA timing includes a
large startup/transfer component and is slower than CPU. Longer recordings
should be used before choosing a default device based on throughput alone.
