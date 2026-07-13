from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from src.data.music_detection import MusicDetectionConfig, run_music_detection_manifest


def load_env_file(path: Path) -> None:
    """
    Подгружает простые KEY=VALUE из .env без дополнительной зависимости.

    Значения из настоящего окружения не перетираем: так на сервере можно задавать
    HF_TOKEN через secrets/env, а локально удобно держать его в .env. Модель
    публичная, токен не обязателен, но пригодится для приватных зеркал.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value



def parse_args() -> argparse.Namespace:
    """Описывает CLI-параметры music-detection stage."""
    parser = argparse.ArgumentParser(description="Detect music/singing in audio manifests.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, default=None)

    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, auto")
    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--music-threshold", type=float, default=0.5)
    parser.add_argument("--top-k-labels", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-threads", type=int, default=1)

    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_env_file(Path(".env"))
    args = parse_args()

    # CLI не содержит model/inference логики: она живет в src/data/music_detection.py,
    # чтобы stage можно было тестировать без загрузки тяжелой AST-модели.
    config = MusicDetectionConfig(
        model_name=args.model_name,
        window_seconds=args.window_seconds,
        music_threshold=args.music_threshold,
        top_k_labels=args.top_k_labels,
        batch_size=args.batch_size,
        max_threads=args.max_threads if args.max_threads > 0 else None,
    )

    outputs = run_music_detection_manifest(
        input_manifest=args.input_manifest,
        output_features=args.output_features,
        output_metadata=args.output_metadata,
        device=args.device,
        config=config,
        hf_token=args.hf_token,
        run_id=args.run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
        fail_fast=args.fail_fast,
    )

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
                    "num_music_flagged": outputs.num_music_flagged,
                    "audio_duration": round(outputs.audio_duration, 3),
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