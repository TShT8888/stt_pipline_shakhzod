from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from src.data import music_detection
from src.data.music_detection import (
    MusicDetectionConfig,
    MusicRuntime,
    build_music_feature_row,
    find_label_index,
    frame_into_windows,
    run_music_detection_manifest,
    summarize_window_scores,
)


FAKE_ID2LABEL = {0: "Speech", 1: "Music", 2: "Silence"}
FAKE_LABEL2ID = {name.lower(): index for index, name in FAKE_ID2LABEL.items()}


def test_find_label_index_is_case_insensitive() -> None:
    assert find_label_index(FAKE_LABEL2ID, "music") == 1
    assert find_label_index(FAKE_LABEL2ID, "Music") == 1
    assert find_label_index(FAKE_LABEL2ID, "Speech") == 0
    assert find_label_index(FAKE_LABEL2ID, "missing") is None


def test_frame_into_windows_pads_last_window() -> None:
    windows = frame_into_windows(np.ones(25, dtype=np.float32), window_samples=10)

    assert windows.shape == (3, 10)
    assert windows[0].tolist() == [1.0] * 10
    assert windows[2][:5].tolist() == [1.0] * 5
    assert windows[2][5:].tolist() == [0.0] * 5


def test_frame_into_windows_short_signal_single_window() -> None:
    windows = frame_into_windows(np.ones(4, dtype=np.float32), window_samples=10)
    assert windows.shape == (1, 10)


def test_summarize_window_scores_flags_music() -> None:
    summary = summarize_window_scores(np.array([0.2, 0.9]), threshold=0.5)

    assert summary["music_probability"] == 0.9
    assert summary["is_music"] is True
    assert summary["music_ratio"] == 0.5


def test_summarize_window_scores_keeps_speech() -> None:
    summary = summarize_window_scores(np.array([0.1, 0.2]), threshold=0.5)

    assert summary["is_music"] is False
    assert summary["music_ratio"] == 0.0


def test_build_music_feature_row_schema(tmp_path) -> None:
    row = {"audio_id": "clip_1", "audio_path": "audio/clip_1.flac", "duration": 6.5, "language": "uz"}
    metrics = {
        "music_probability": 0.92,
        "music_mean_probability": 0.5,
        "music_ratio": 1.0,
        "is_music": True,
        "num_windows": 1,
        "top_labels": [{"label": "Music", "probability": 0.92}],
    }
    feature_row = build_music_feature_row(
        source_row=row,
        input_manifest=tmp_path / "manifest.jsonl",
        line_number=1,
        audio_path=tmp_path / "audio" / "clip_1.flac",
        metrics=metrics,
        run_id="music_test",
        config=MusicDetectionConfig(),
    )

    assert feature_row["audio_id"] == "clip_1"
    assert feature_row["feature_type"] == "music_detection"
    assert feature_row["status"] == "ok"
    assert feature_row["is_music"] is True
    assert feature_row["music_probability"] == 0.92


def _fake_runtime() -> MusicRuntime:
    return MusicRuntime(
        model=None,
        feature_extractor=None,
        id2label=FAKE_ID2LABEL,
        label2id=FAKE_LABEL2ID,
        music_index=1,
    )


def test_run_music_detection_manifest_end_to_end(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.jsonl"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "clip_1.wav", np.zeros(1600, dtype=np.float32), 16000)
    manifest.write_text(
        json.dumps(
            {"audio_id": "clip_1", "audio_path": "audio/clip_1.wav", "duration": 0.1, "language": "uz"}
        )
        + "\n",
        encoding="utf-8",
    )

    # Подменяем загрузку модели и инференс: тест остается быстрым и офлайн.
    monkeypatch.setattr(music_detection, "load_music_runtime", lambda *a, **k: _fake_runtime())
    monkeypatch.setattr(
        music_detection,
        "score_windows",
        lambda runtime, windows, device, config: np.array([[0.1, 0.95, 0.05]]),
    )

    output_features = tmp_path / "music.jsonl"
    output_metadata = tmp_path / "music_metadata.json"
    outputs = run_music_detection_manifest(
        input_manifest=manifest,
        output_features=output_features,
        output_metadata=output_metadata,
        device="cpu",
        config=MusicDetectionConfig(),
        limit=1,
    )

    assert outputs.num_processed_rows == 1
    assert outputs.num_errors == 0
    assert outputs.num_music_flagged == 1

    rows = [json.loads(line) for line in output_features.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["is_music"] is True
    assert rows[0]["music_probability"] == 0.95
    assert rows[0]["top_labels"][0]["label"] == "Music"
    assert output_metadata.exists()