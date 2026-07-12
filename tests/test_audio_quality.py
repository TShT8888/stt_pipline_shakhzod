from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from src.data.audio_quality import (
    AudioQualityConfig,
    compute_audio_quality_metrics,
    run_audio_quality_manifest,
)


def test_compute_audio_quality_metrics_for_clean_sine() -> None:
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1).astype(np.float32)

    metrics = compute_audio_quality_metrics(audio, sample_rate, AudioQualityConfig())

    assert metrics["duration"] == 1.0
    assert metrics["sample_rate"] == sample_rate
    assert metrics["channels"] == 1
    assert metrics["rms_dbfs"] < -20.0
    assert metrics["peak_dbfs"] == -20.0
    assert metrics["clipping_ratio"] == 0.0
    assert metrics["quality_score"] > 0.5


def test_compute_audio_quality_metrics_detects_clipping() -> None:
    audio = np.ones((100, 1), dtype=np.float32)
    metrics = compute_audio_quality_metrics(audio, 100, AudioQualityConfig())

    assert metrics["clipping_ratio"] == 1.0
    assert metrics["quality_score"] < 0.7


def test_run_audio_quality_manifest_writes_feature_rows(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "sample.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    manifest.write_text(
        json.dumps(
            {
                "audio_id": "sample_001",
                "audio_path": "audio/sample.wav",
                "duration": 0.1,
                "dataset": "example/dataset",
                "language": "uz",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_features = tmp_path / "quality.jsonl"
    output_metadata = tmp_path / "quality_metadata.json"
    outputs = run_audio_quality_manifest(
        input_manifest=manifest,
        output_features=output_features,
        output_metadata=output_metadata,
        config=AudioQualityConfig(),
        limit=1,
    )

    assert outputs.num_processed_rows == 1
    assert outputs.num_errors == 0
    rows = [json.loads(line) for line in output_features.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["audio_id"] == "sample_001"
    assert rows[0]["feature_type"] == "audio_quality"
    assert rows[0]["status"] == "ok"
    assert rows[0]["duration"] == 0.1
    assert output_metadata.exists()
