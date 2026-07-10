from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.data.vad import VadConfig, run_vad_manifest, runtime_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAD over unlabeled audio manifest.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("data/raw/unlabeled/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-segments",
        type=Path,
        default=Path("data/interim/vad/unlabeled_segments.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("data/interim/vad/unlabeled_audio_summary.jsonl"),
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("data/interim/vad/unlabeled_run_metadata.json"),
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--min-silence-duration-ms", type=int, default=300)
    parser.add_argument("--speech-pad-ms", type=int, default=100)
    parser.add_argument("--min-segment-duration", type=float, default=0.5)
    parser.add_argument("--max-segment-duration", type=float, default=30.0)
    parser.add_argument(
        "--max-threads",
        type=int,
        default=1,
        help="Torch CPU threads for predictable preprocessing throughput. Use 0 to keep default.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--vad-run-id", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VadConfig(
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        min_segment_duration=args.min_segment_duration,
        max_segment_duration=args.max_segment_duration,
        max_threads=args.max_threads if args.max_threads > 0 else None,
    )

    outputs = run_vad_manifest(
        input_manifest=args.input_manifest,
        output_segments=args.output_segments,
        output_summary=args.output_summary,
        output_metadata=args.output_metadata,
        device=args.device,
        config=config,
        vad_run_id=args.vad_run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        fail_fast=args.fail_fast,
    )

    report = {
        "status": "ok",
        "outputs": {
            "segments_path": str(outputs.segments_path),
            "summary_path": str(outputs.summary_path),
            "metadata_path": str(outputs.metadata_path),
            "num_input_rows": outputs.num_input_rows,
            "num_processed_rows": outputs.num_processed_rows,
            "num_skipped_shard_rows": outputs.num_skipped_shard_rows,
            "num_segments": outputs.num_segments,
            "num_summaries": outputs.num_summaries,
            "num_errors": outputs.num_errors,
            "audio_duration": round(outputs.audio_duration, 3),
            "processing_seconds": round(outputs.processing_seconds, 3),
            "real_time_factor": (
                round(outputs.real_time_factor, 6)
                if outputs.real_time_factor is not None
                else None
            ),
        },
        "runtime": runtime_metadata(args.device, config),
        "config": asdict(config),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
