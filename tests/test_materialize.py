from pathlib import Path

import numpy as np

from src.data.materialize import (
    MaterializeConfig,
    build_materialized_row,
    output_audio_path,
    slice_audio,
)


def test_output_audio_path() -> None:
    path, relpath = output_audio_path(Path("out"), "segment_001", "flac")

    assert path == Path("out/audio/segment_001.flac")
    assert relpath == "audio/segment_001.flac"


def test_slice_audio_uses_seconds() -> None:
    audio = np.arange(20, dtype=np.float32).reshape(10, 2)

    clip = slice_audio(audio, sample_rate=10, start_sec=0.2, end_sec=0.5)

    assert clip.tolist() == audio[2:5].tolist()


def test_slice_audio_returns_empty_for_invalid_range() -> None:
    audio = np.arange(20, dtype=np.float32).reshape(10, 2)

    clip = slice_audio(audio, sample_rate=10, start_sec=0.8, end_sec=0.2)

    assert clip.shape == (0, 2)


def test_build_materialized_row_preserves_provenance() -> None:
    segment = {
        "segment_id": "seg_001",
        "source_audio_id": "src_001",
        "source_audio_path": "data/raw/audio.wav",
        "source_audio_relpath": "audio.wav",
        "source_manifest": "manifest.jsonl",
        "source_line_number": 3,
        "vad_start": 1.2,
        "vad_end": 4.5,
        "vad_duration": 3.3,
        "vad_run_id": "vad_abc",
        "vad_model": "silero-vad",
        "vad_threshold": 0.5,
        "language": "uz",
        "dataset": "example",
    }

    row = build_materialized_row(
        segment=segment,
        audio_relpath="audio/seg_001.flac",
        sample_rate=16000,
        channels=1,
        actual_duration=3.299,
    )

    assert row["audio_id"] == "seg_001"
    assert row["audio_path"] == "audio/seg_001.flac"
    assert row["duration"] == 3.299
    assert row["source_audio_id"] == "src_001"
    assert row["source_start"] == 1.2
    assert row["source_end"] == 4.5
    assert row["vad_run_id"] == "vad_abc"


def test_materialize_config_rejects_unknown_format_later() -> None:
    config = MaterializeConfig(output_format="mp3")

    assert config.output_format == "mp3"
