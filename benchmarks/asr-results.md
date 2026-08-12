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
is faster.

## Machine

| Date | Windows | CPU | RAM | GPU | Driver | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-12 | Windows 11 10.0.26200 | Intel64 Family 6 Model 158, 6 logical processors | 23.9 GB | NVIDIA GeForce GTX 1650 SUPER, 4096 MiB | 596.36 | Official whisper.cpp v1.8.5 cuBLAS 12.4 bundle; CUDA Toolkit/nvcc not installed |

## Results

| Backend | Model | Device | Audio | Elapsed (s) | RTF | WER | Segments | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Whisper.cpp | ggml-small.bin | cpu | videorc-session.wav (19.200 s) | 40.499 | 2.109 | — | 4 | CUDA build with `--no-gpu` |
| Whisper.cpp | ggml-small.bin | cuda | videorc-session.wav (19.200 s) | 71.320 | 3.715 | — | 4 | CUDA device 0 active; short clip includes model/GPU startup overhead |

Peak RAM/VRAM and qualitative timestamp/punctuation observations should be
recorded in the notes column or below after each run. The first scaffold keeps
the benchmark dependency-free and reports wall time, RTF, segment count, and
WER; it does not silently claim process-level peak memory measurements.

## Installation notes

The selected runtime is the native [`parakeet-cli`](https://github.com/ggml-org/whisper.cpp/tree/master/examples/parakeet-cli)
target from whisper.cpp, using the Q8_0 model published in
[`ggml-org/parakeet-GGUF`](https://huggingface.co/ggml-org/parakeet-GGUF).
See the upstream references before downloading or building the optional
runtime.

Current local status: Parakeet runtime and model are not installed.

The current CUDA build initializes the GTX 1650 SUPER successfully. The
recorded clip is short and uses the small model, so its CUDA timing includes a
large startup/transfer component and is slower than CPU. Longer recordings
should be used before choosing a default device based on throughput alone.
