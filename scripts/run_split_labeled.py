from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.data.labeled_split import LabeledSplitConfig, SplitRatios, run_labeled_split
from src.data.selection import QualityGates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter labeled speech and split train/val/test.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--quality-features", type=Path, nargs="+", required=True)
    parser.add_argument("--overlap-features", type=Path, nargs="*", default=[])
    parser.add_argument("--music-features", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--group-key", default="audio_id")
    parser.add_argument("--stratify-keys", nargs="*", default=["dataset", "language"])
    parser.add_argument("--salt", default="labeled_split_v1")

    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--max-clipping-ratio", type=float, default=0.001)
    parser.add_argument("--max-silence-ratio", type=float, default=0.6)
    parser.add_argument("--min-rms-dbfs", type=float, default=-45.0)
    parser.add_argument("--max-rms-dbfs", type=float, default=-3.0)
    parser.add_argument("--min-snr-db", type=float, default=5.0)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.2)
    parser.add_argument("--max-dc-offset", type=float, default=0.01)
    parser.add_argument("--keep-music", action="store_true")
    parser.add_argument("--no-require-text", action="store_true")
    parser.add_argument("--min-chars-per-second", type=float, default=3.0)
    parser.add_argument("--max-chars-per-second", type=float, default=25.0)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"train+val+test ratios must sum to 1.0, got {ratio_sum}")

    gates = QualityGates(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        target_sample_rate=args.target_sample_rate,
        max_clipping_ratio=args.max_clipping_ratio,
        max_silence_ratio=args.max_silence_ratio,
        min_rms_dbfs=args.min_rms_dbfs,
        max_rms_dbfs=args.max_rms_dbfs,
        min_snr_db=args.min_snr_db,
        max_overlap_ratio=args.max_overlap_ratio,
        max_dc_offset=args.max_dc_offset,
        drop_music=not args.keep_music,
        require_overlap=bool(args.overlap_features),
        require_music=bool(args.music_features),
    )
    config = LabeledSplitConfig(
        gates=gates,
        ratios=SplitRatios(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio),
        group_key=args.group_key,
        stratify_keys=tuple(args.stratify_keys),
        require_text=not args.no_require_text,
        min_chars_per_second=args.min_chars_per_second,
        max_chars_per_second=args.max_chars_per_second,
        allowed_languages=tuple(args.languages) if args.languages else None,
        salt=args.salt,
    )
    outputs = run_labeled_split(
        input_manifest=args.input_manifest,
        quality_features=args.quality_features,
        overlap_features=args.overlap_features,
        music_features=args.music_features,
        output_dir=args.output_dir,
        config=config,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "outputs": {
                    "output_dir": str(outputs.output_dir),
                    "metadata_path": str(outputs.metadata_path),
                    "num_input_rows": outputs.num_input_rows,
                    "num_selected": outputs.num_selected,
                    "num_rejected": outputs.num_rejected,
                    "split_counts": outputs.split_counts,
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