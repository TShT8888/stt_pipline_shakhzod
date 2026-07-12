from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.data.audio_quality import AudioQualityConfig, run_audio_quality_manifest


def parse_args() -> argparse.Namespace:
    """Описывает CLI-параметры без бизнес-логики."""
    parser = argparse.ArgumentParser(description="Compute audio quality features for manifests.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, default=None)

    parser.add_argument("--silence-threshold-dbfs", type=float, default=-50.0)
    parser.add_argument("--clipping-threshold", type=float, default=0.999)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--noise-percentile", type=float, default=10.0)
    parser.add_argument("--target-min-dbfs", type=float, default=-35.0)
    parser.add_argument("--target-max-dbfs", type=float, default=-8.0)

    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # CLI отвечает только за сбор параметров. Все расчеты качества находятся в
    # src/data/audio_quality.py, чтобы stage можно было переиспользовать из кода.
    config = AudioQualityConfig(
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        clipping_threshold=args.clipping_threshold,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        noise_percentile=args.noise_percentile,
        target_min_dbfs=args.target_min_dbfs,
        target_max_dbfs=args.target_max_dbfs,
    )

    outputs = run_audio_quality_manifest(
        input_manifest=args.input_manifest,
        output_features=args.output_features,
        output_metadata=args.output_metadata,
        config=config,
        run_id=args.run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
        fail_fast=args.fail_fast,
    )

    # Печатаем короткий JSON-отчет в stdout, чтобы одинаково удобно запускать
    # локально, на сервере, в notebook или внутри batch-job.
    print(
        json.dumps(
            {
                "status": "ok",
                "outputs": {
                    "features_path": str(outputs.features_path),
                    "metadata_path": (
                        str(outputs.metadata_path) if outputs.metadata_path is not None else None
                    ),
                    "num_input_rows": outputs.num_input_rows,
                    "num_processed_rows": outputs.num_processed_rows,
                    "num_skipped_shard_rows": outputs.num_skipped_shard_rows,
                    "num_errors": outputs.num_errors,
                    "audio_duration": round(outputs.audio_duration, 3),
                    "mean_quality_score": (
                        round(outputs.mean_quality_score, 6)
                        if outputs.mean_quality_score is not None
                        else None
                    ),
                    "processing_seconds": round(outputs.processing_seconds, 3),
                },
                "config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
