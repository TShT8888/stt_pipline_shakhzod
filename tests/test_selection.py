from __future__ import annotations

import json

from src.data.labeled_split import (
    LabeledSplitConfig,
    SplitRatios,
    assign_splits,
    evaluate_text_gates,
    run_labeled_split,
)
from src.data.selection import QualityGates, evaluate_quality_gates, selection_score, stable_unit_hash
from src.data.ssl_selection import SslSelectionConfig, run_ssl_selection


GOOD_QUALITY = {
    "status": "ok", "clipping_ratio": 0.0, "silence_ratio": 0.1, "rms_dbfs": -20.0,
    "snr_estimate_db": 20.0, "dc_offset_mean_abs": 0.0, "quality_score": 0.9,
}
GOOD_OVERLAP = {"status": "ok", "overlap_ratio": 0.0}
GOOD_MUSIC = {"status": "ok", "is_music": False}
SSL_GATES = QualityGates(min_duration=3.0, max_duration=30.0)


def test_gates_pass_for_clean_clip() -> None:
    reasons = evaluate_quality_gates(
        duration=6.0, sample_rate=16000, quality=GOOD_QUALITY, overlap=GOOD_OVERLAP,
        music=GOOD_MUSIC, gates=SSL_GATES,
    )
    assert reasons == []


def test_gates_collect_all_failures() -> None:
    bad_quality = {**GOOD_QUALITY, "clipping_ratio": 0.5, "snr_estimate_db": 2.0}
    reasons = evaluate_quality_gates(
        duration=1.0, sample_rate=8000, quality=bad_quality,
        overlap={"status": "ok", "overlap_ratio": 0.9}, music={"status": "ok", "is_music": True},
        gates=SSL_GATES,
    )
    assert "duration_too_short" in reasons
    assert "sample_rate_mismatch" in reasons
    assert "clipping" in reasons
    assert "low_snr" in reasons
    assert "overlap" in reasons
    assert "music" in reasons


def test_missing_features_are_flagged() -> None:
    reasons = evaluate_quality_gates(
        duration=6.0, sample_rate=16000, quality=None, overlap=None, music=None, gates=SSL_GATES,
    )
    assert {"missing_quality", "missing_overlap", "missing_music"} <= set(reasons)


def test_selection_score_prefers_clean_low_overlap() -> None:
    high = selection_score(GOOD_QUALITY, GOOD_OVERLAP)
    low = selection_score(
        {**GOOD_QUALITY, "quality_score": 0.5}, {"status": "ok", "overlap_ratio": 0.3}
    )
    assert high > low


def test_stable_unit_hash_is_deterministic_and_bounded() -> None:
    assert stable_unit_hash("abc", "salt") == stable_unit_hash("abc", "salt")
    assert 0.0 <= stable_unit_hash("abc", "salt") < 1.0
    assert stable_unit_hash("abc", "salt") != stable_unit_hash("abc", "other")


def test_text_gates() -> None:
    config = LabeledSplitConfig()
    assert evaluate_text_gates(None, 4.0, config) == ["missing_text"]
    assert evaluate_text_gates("!!!", 4.0, config) == ["empty_text"]
    assert evaluate_text_gates("salom dunyo qandaysiz", 4.0, config) == []
    assert "text_too_dense" in evaluate_text_gates("a" * 200, 4.0, config)


def test_assign_splits_keeps_groups_together_and_ratios() -> None:
    records = [
        {"audio_id": f"id_{i}", "speaker": f"spk_{i // 4}", "dataset": "d", "language": "uz"}
        for i in range(40)
    ]
    assignment = assign_splits(
        records, SplitRatios(0.8, 0.1, 0.1), group_key="speaker",
        stratify_keys=("dataset", "language"), salt="t",
    )
    # все клипы одного спикера в одном сплите (нет утечки)
    for speaker_start in range(0, 40, 4):
        splits = {assignment[f"id_{i}"] for i in range(speaker_start, speaker_start + 4)}
        assert len(splits) == 1
    assert set(assignment.values()) <= {"train", "val", "test"}


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_run_ssl_selection_end_to_end(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [
        {"audio_id": "a", "audio_path": "audio/a.flac", "duration": 6.0, "sample_rate": 16000,
         "language": "uz", "source_audio_id": "src1"},
        {"audio_id": "b", "audio_path": "audio/b.flac", "duration": 6.0, "sample_rate": 16000,
         "language": "uz", "source_audio_id": "src1"},  # музыка -> отсев
    ])
    quality = tmp_path / "quality.jsonl"
    _write_jsonl(quality, [{"audio_id": "a", **GOOD_QUALITY}, {"audio_id": "b", **GOOD_QUALITY}])
    overlap = tmp_path / "overlap.jsonl"
    _write_jsonl(overlap, [{"audio_id": "a", **GOOD_OVERLAP}, {"audio_id": "b", **GOOD_OVERLAP}])
    music = tmp_path / "music.jsonl"
    _write_jsonl(music, [
        {"audio_id": "a", "status": "ok", "is_music": False},
        {"audio_id": "b", "status": "ok", "is_music": True},
    ])

    outputs = run_ssl_selection(
        input_manifest=manifest, quality_features=[quality], overlap_features=[overlap],
        music_features=[music], output_selected=tmp_path / "selected.jsonl",
        output_rejected=tmp_path / "rejected.jsonl", output_metadata=tmp_path / "meta.json",
        config=SslSelectionConfig(),
    )
    assert outputs.num_selected == 1
    assert outputs.num_rejected == 1
    selected = [json.loads(x) for x in (tmp_path / "selected.jsonl").read_text().splitlines()]
    assert selected[0]["audio_id"] == "a"
    assert "selection_score" in selected[0]


def test_run_labeled_split_end_to_end(tmp_path) -> None:
    rows = [
        {"audio_id": f"c{i}", "audio_path": f"audio/c{i}.wav", "duration": 4.0,
         "sample_rate": 16000, "language": "uz", "dataset": "d", "text": "salom dunyo bugun"}
        for i in range(20)
    ]
    manifest = tmp_path / "labeled.jsonl"
    _write_jsonl(manifest, rows)
    quality = tmp_path / "quality.jsonl"
    _write_jsonl(quality, [{"audio_id": f"c{i}", **GOOD_QUALITY} for i in range(20)])

    gates = QualityGates(
        min_duration=0.8, max_duration=30.0, max_silence_ratio=0.6, min_snr_db=5.0,
        max_overlap_ratio=0.2, require_overlap=False, require_music=False,
    )
    outputs = run_labeled_split(
        input_manifest=manifest, quality_features=[quality], overlap_features=[], music_features=[],
        output_dir=tmp_path / "splits", config=LabeledSplitConfig(gates=gates),
    )
    assert outputs.num_selected == 20
    assert sum(outputs.split_counts.values()) == 20
    assert (tmp_path / "splits" / "train.jsonl").exists()
    assert (tmp_path / "splits" / "val.jsonl").exists()
    assert (tmp_path / "splits" / "test.jsonl").exists()
    assert outputs.split_counts["train"] >= outputs.split_counts["val"]
