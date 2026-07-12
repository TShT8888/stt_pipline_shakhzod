from __future__ import annotations

import json
from pathlib import Path

from src.data.overlap import (
    OverlapConfig,
    build_overlap_feature_row,
    build_overlap_run_id,
    resolve_audio_path,
    run_overlap_manifest,
)


class FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    def itertracks(self, *, yield_label: bool) -> list[tuple[FakeSegment, None, str]]:
        assert yield_label is True
        return [
            (FakeSegment(0.1, 0.4), None, "overlap"),
            (FakeSegment(1.0, 1.05), None, "overlap"),
        ]


class FakePipeline:
    def __call__(self, audio: dict) -> FakeAnnotation:
        assert set(audio) == {"waveform", "sample_rate"}
        assert audio["sample_rate"] == 16000
        return FakeAnnotation()


def test_resolve_audio_path_relative_to_manifest() -> None:
    manifest = Path("data/raw/labeled/metadata.jsonl")
    assert resolve_audio_path(manifest, {"audio_path": "audio/example.wav"}) == Path(
        "data/raw/labeled/audio/example.wav"
    )


def test_build_overlap_feature_row_uses_project_manifest_schema() -> None:
    row = {
        "audio_id": "sample_001",
        "audio_path": "audio/sample_001.wav",
        "duration": 10.0,
        "dataset": "example/dataset",
        "language": "uz",
        "split": "train",
    }

    feature = build_overlap_feature_row(
        source_row=row,
        input_manifest=Path("data/raw/labeled/metadata.jsonl"),
        line_number=3,
        audio_path=Path("data/raw/labeled/audio/sample_001.wav"),
        overlap_segments=[{"start": 1.0, "end": 2.0, "duration": 1.0}],
        run_id="overlap_test",
        config=OverlapConfig(),
    )

    assert feature["audio_id"] == "sample_001"
    assert feature["feature_type"] == "overlap"
    assert feature["status"] == "ok"
    assert feature["duration"] == 10.0
    assert feature["overlap_duration"] == 1.0
    assert feature["overlap_ratio"] == 0.1
    assert feature["num_overlap_segments"] == 1
    assert feature["dataset"] == "example/dataset"
    assert feature["language"] == "uz"
    assert feature["split"] == "train"


def test_build_overlap_run_id_changes_with_limit() -> None:
    config = OverlapConfig()
    assert build_overlap_run_id(Path("manifest.jsonl"), config, 0, 1, 10) != build_overlap_run_id(
        Path("manifest.jsonl"), config, 0, 1, 20
    )


def test_run_overlap_manifest_writes_feature_rows(tmp_path, monkeypatch) -> None:
    import src.data.overlap as overlap_module

    manifest = tmp_path / "manifest.jsonl"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "audio_id": "sample_001",
                "audio_path": "audio/example.wav",
                "duration": 2.0,
                "dataset": "example/dataset",
                "language": "uz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import soundfile as sf

    sf.write(audio_dir / "example.wav", [0.0, 0.0, 0.0], 16000)

    monkeypatch.setattr(
        overlap_module,
        "load_overlap_pipeline",
        lambda config, device, hf_token: FakePipeline(),
    )

    output_features = tmp_path / "features.jsonl"
    output_metadata = tmp_path / "metadata.json"
    outputs = run_overlap_manifest(
        input_manifest=manifest,
        output_features=output_features,
        output_metadata=output_metadata,
        device="cpu",
        config=OverlapConfig(min_overlap_duration=0.1),
        limit=1,
    )

    assert outputs.num_processed_rows == 1
    assert outputs.num_errors == 0
    rows = [json.loads(line) for line in output_features.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["audio_id"] == "sample_001"
    assert rows[0]["overlap_duration"] == 0.3
    assert rows[0]["num_overlap_segments"] == 1
    assert output_metadata.exists()
