from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.materialize import MaterializeConfig, materialize_vad_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize VAD segments into audio clips.")
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("data/interim/vad/unlabeled_segments.jsonl"),
        help="VAD segments JSONL produced by run_vad_unlabeled.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/unlabeled_vad"),
    )
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--output-metadata", type=Path, default=None)
    parser.add_argument("--format", choices=["flac", "wav"], default="flac")
    parser.add_argument(
        "--subtype",
        default=None,
        help="Optional soundfile subtype, e.g. PCM_16 for wav. Default keeps soundfile default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MaterializeConfig(
        output_format=args.format,
        subtype=args.subtype,
        overwrite=args.overwrite,
    )
    outputs = materialize_vad_segments(
        segments_path=args.segments,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        output_metadata=args.output_metadata,
        config=config,
        fail_fast=args.fail_fast,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "outputs": {
                    "manifest_path": str(outputs.manifest_path),
                    "metadata_path": (
                        str(outputs.metadata_path) if outputs.metadata_path is not None else None
                    ),
                    "num_input_segments": outputs.num_input_segments,
                    "num_written_segments": outputs.num_written_segments,
                    "num_skipped_existing": outputs.num_skipped_existing,
                    "num_errors": outputs.num_errors,
                    "audio_duration": round(outputs.audio_duration, 3),
                    "processing_seconds": round(outputs.processing_seconds, 3),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
