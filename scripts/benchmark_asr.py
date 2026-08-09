"""Run a local ASR benchmark and emit one JSON record per audio file."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from video_mcp.config import load_config
from video_mcp.evaluation import benchmark_asr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one configured local ASR backend on normalized WAV files."
    )
    parser.add_argument("audio", nargs="+", type=Path, help="16 kHz mono WAV files.")
    parser.add_argument("--config", type=Path, help="Path to video-mcp.yaml.")
    parser.add_argument(
        "--backend",
        choices=("whisper_cpp", "parakeet"),
        help="Override the backend from configuration.",
    )
    parser.add_argument("--model", type=Path, help="Override the configured model path.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override the configured ASR device.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Optional directory of UTF-8 .txt references matching each WAV stem.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    asr = config.asr
    if args.backend or args.model:
        asr = replace(
            asr,
            backend=args.backend or asr.backend,
            model=(args.model.expanduser().resolve() if args.model else asr.model),
        )
        config = replace(config, asr=asr)

    for audio in args.audio:
        reference = None
        if args.reference_dir:
            reference_path = args.reference_dir / f"{audio.stem}.txt"
            if reference_path.is_file():
                reference = reference_path.read_text(encoding="utf-8")
        result = benchmark_asr(
            audio,
            config,
            reference_text=reference,
            device=args.device,
            threads=args.threads,
        )
        print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
