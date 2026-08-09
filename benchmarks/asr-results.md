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
| | | | | | | |

## Results

| Backend | Model | Device | Audio | Elapsed (s) | RTF | WER | Segments | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Whisper.cpp | | | | | | | | |
| Parakeet | | | | | | | | |

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
