from pathlib import Path

import pytest

from src.data.vad import (
    VadConfig,
    build_segment_row,
    build_vad_run_id,
    resolve_audio_path,
    split_long_segment,
)


def test_split_long_segment_keeps_short_segment() -> None:
    assert split_long_segment(1.0, 10.0, max_duration=30.0) == [(1.0, 10.0)]


def test_split_long_segment_splits_long_segment() -> None:
    assert split_long_segment(0.0, 65.0, max_duration=30.0) == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 65.0),
    ]


def test_split_long_segment_rejects_invalid_max_duration() -> None:
    with pytest.raises(ValueError, match="max_duration must be positive"):
        split_long_segment(0.0, 1.0, max_duration=0.0)


def test_resolve_audio_path_relative_to_manifest() -> None:
    manifest = Path("src/data/raw/unlabeled/manifest.jsonl")
    assert resolve_audio_path(manifest, "audio/example.wav") == Path(
        "src/data/raw/unlabeled/audio/example.wav"
    )


def test_build_segment_row_preserves_source_metadata() -> None:
    row = {
        "audio_id": "unlabeled_001",
        "audio_path": "audio/unlabeled_001.wav",
        "dataset": "example/dataset",
        "language": "uz",
        "duration": 100.0,
        "sample_rate": 16000,
        "channels": 1,
        "source_clips": 20,
    }

    segment = build_segment_row(
        source_row=row,
        input_manifest=Path("src/data/raw/unlabeled/manifest.jsonl"),
        source_audio_path=Path("src/data/raw/unlabeled/audio/unlabeled_001.wav"),
        line_number=7,
        vad_index=3,
        part_index=1,
        start_sec=10.12345,
        end_sec=12.98765,
        config=VadConfig(threshold=0.7),
        vad_run_id="vad_test",
    )

    assert segment["segment_id"] == "unlabeled_001_vad_00003_01"
    assert segment["source_audio_id"] == "unlabeled_001"
    assert segment["source_line_number"] == 7
    assert segment["dataset"] == "example/dataset"
    assert segment["language"] == "uz"
    assert segment["vad_start"] == 10.123
    assert segment["vad_end"] == 12.988
    assert segment["vad_duration"] == 2.864
    assert segment["vad_run_id"] == "vad_test"
    assert segment["vad_threshold"] == 0.7


def test_build_vad_run_id_is_stable() -> None:
    config = VadConfig(threshold=0.7)
    assert build_vad_run_id(Path("manifest.jsonl"), config, 0, 4) == build_vad_run_id(
        Path("manifest.jsonl"), config, 0, 4
    )
    assert build_vad_run_id(Path("manifest.jsonl"), config, 1, 4) != build_vad_run_id(
        Path("manifest.jsonl"), config, 0, 4
    )
