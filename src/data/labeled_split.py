from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data.jsonl import JsonObject, JsonlWriter, read_jsonl, write_json
from src.data.selection import (
    QualityGates,
    build_run_id,
    evaluate_quality_gates,
    load_feature_index,
    stable_unit_hash,
)
from src.data.text_normalization import normalize_text


SPLITS = ("train", "val", "test")


def default_labeled_gates() -> QualityGates:
    """Пороги для labeled: мягче, чем SSL, — размеченные данные ценнее."""
    return QualityGates(
        min_duration=0.8,
        max_duration=30.0,
        max_silence_ratio=0.6,
        min_snr_db=5.0,
        max_overlap_ratio=0.2,
    )


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


@dataclass(frozen=True)
class LabeledSplitConfig:
    """Отбор размеченных клипов + детерминированный сплит train/val/test."""

    gates: QualityGates = field(default_factory=default_labeled_gates)
    ratios: SplitRatios = field(default_factory=SplitRatios)
    group_key: str = "audio_id"
    stratify_keys: tuple[str, ...] = ("dataset", "language")
    require_text: bool = True
    min_chars_per_second: float = 3.0
    max_chars_per_second: float = 25.0
    allowed_languages: tuple[str, ...] | None = None
    salt: str = "labeled_split_v1"


@dataclass(frozen=True)
class LabeledSplitOutputs:
    output_dir: Path
    metadata_path: Path | None
    num_input_rows: int
    num_selected: int
    num_rejected: int
    split_counts: dict[str, int]
    processing_seconds: float


def evaluate_text_gates(text: object, duration: float | None, config: LabeledSplitConfig) -> list[str]:
    """Проверяет наличие и правдоподобность транскрипта (символов в секунду)."""
    if not config.require_text:
        return []
    if text is None:
        return ["missing_text"]

    normalized = normalize_text(str(text))
    if not normalized:
        return ["empty_text"]

    reasons: list[str] = []
    if duration and duration > 0:
        chars_per_second = len(normalized) / duration
        if chars_per_second < config.min_chars_per_second:
            reasons.append("text_too_sparse")
        if chars_per_second > config.max_chars_per_second:
            reasons.append("text_too_dense")
    return reasons


def assign_splits(
    records: Sequence[JsonObject],
    ratios: SplitRatios,
    group_key: str,
    stratify_keys: Sequence[str],
    salt: str,
) -> dict[str, str]:
    """
    Детерминированно раскидывает записи по train/val/test.

    Внутри каждой страты (по `stratify_keys`) записи группируются по `group_key`,
    группы сортируются по стабильному хэшу и целиком уходят в один сплит — так нет
    утечки диктора между сплитами, а пропорции соблюдаются по числу клипов.
    """
    assignment: dict[str, str] = {}
    strata: dict[tuple[str, ...], list[JsonObject]] = defaultdict(list)
    for record in records:
        stratum = tuple(str(record.get(key)) for key in stratify_keys)
        strata[stratum].append(record)

    train_end = ratios.train
    val_end = ratios.train + ratios.val
    for stratum_records in strata.values():
        groups: dict[str, list[JsonObject]] = defaultdict(list)
        for record in stratum_records:
            groups[str(record.get(group_key))].append(record)

        ordered = sorted(groups.items(), key=lambda kv: (stable_unit_hash(kv[0], salt), kv[0]))
        total = float(sum(len(recs) for _, recs in ordered))
        accumulated = 0
        for _group_value, group_records in ordered:
            midpoint = (accumulated + accumulated + len(group_records)) / 2.0 / total
            accumulated += len(group_records)
            if midpoint < train_end:
                split = "train"
            elif midpoint < val_end:
                split = "val"
            else:
                split = "test"
            for record in group_records:
                assignment[str(record["audio_id"])] = split
    return assignment


def run_labeled_split(
    *,
    input_manifest: Path,
    quality_features: Sequence[Path],
    overlap_features: Sequence[Path],
    music_features: Sequence[Path],
    output_dir: Path,
    config: LabeledSplitConfig,
    run_id: str | None = None,
) -> LabeledSplitOutputs:
    started_at = time.perf_counter()
    if run_id is None:
        run_id = build_run_id("split", input_manifest, config)

    quality_index = load_feature_index(quality_features)
    overlap_index = load_feature_index(overlap_features)
    music_index = load_feature_index(music_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    num_input_rows = 0
    num_rejected = 0
    reject_histogram: dict[str, int] = defaultdict(int)
    passing: list[JsonObject] = []

    with JsonlWriter(output_dir / "rejected.jsonl") as rejected_writer:
        for _line_number, row in read_jsonl(input_manifest):
            num_input_rows += 1
            audio_id = str(row.get("audio_id"))
            duration = row.get("duration")
            duration_value = float(duration) if duration is not None else None
            reasons = evaluate_quality_gates(
                duration=duration_value,
                sample_rate=row.get("sample_rate"),
                quality=quality_index.get(audio_id),
                overlap=overlap_index.get(audio_id),
                music=music_index.get(audio_id),
                gates=config.gates,
            )
            reasons += evaluate_text_gates(row.get("text"), duration_value, config)
            language = row.get("language")
            if config.allowed_languages is not None and language not in config.allowed_languages:
                reasons.append("language")

            if reasons:
                num_rejected += 1
                for reason in reasons:
                    reject_histogram[reason] += 1
                rejected_writer.write(
                    {
                        "audio_id": audio_id,
                        "reject_reasons": reasons,
                        "duration": duration,
                        "language": language,
                        "dataset": row.get("dataset"),
                    }
                )
                continue

            passing.append(dict(row))

    assignment = assign_splits(
        passing, config.ratios, config.group_key, config.stratify_keys, config.salt
    )

    split_counts: dict[str, int] = {split: 0 for split in SPLITS}
    split_hours: dict[str, float] = {split: 0.0 for split in SPLITS}
    split_language: dict[str, dict[str, int]] = {split: defaultdict(int) for split in SPLITS}

    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(JsonlWriter(output_dir / f"{split}.jsonl"))
            for split in SPLITS
        }
        for record in sorted(passing, key=lambda r: str(r["audio_id"])):
            split = assignment[str(record["audio_id"])]
            out_row = dict(record)
            out_row["split"] = split
            out_row["split_run_id"] = run_id
            writers[split].write(out_row)
            split_counts[split] += 1
            split_hours[split] += float(record.get("duration") or 0.0) / 3600.0
            split_language[split][str(record.get("language"))] += 1

    processing_seconds = time.perf_counter() - started_at
    if run_id and (output_dir / "split_metadata.json"):
        write_json(
            output_dir / "split_metadata.json",
            {
                "split_run_id": run_id,
                "input_manifest": str(input_manifest),
                "output_dir": str(output_dir),
                "num_input_rows": num_input_rows,
                "num_selected": len(passing),
                "num_rejected": num_rejected,
                "split_counts": split_counts,
                "split_hours": {k: round(v, 4) for k, v in split_hours.items()},
                "split_by_language": {k: dict(sorted(v.items())) for k, v in split_language.items()},
                "reject_histogram": dict(sorted(reject_histogram.items())),
                "processing_seconds": round(processing_seconds, 3),
                "config": asdict(config),
            },
        )

    return LabeledSplitOutputs(
        output_dir=output_dir,
        metadata_path=output_dir / "split_metadata.json",
        num_input_rows=num_input_rows,
        num_selected=len(passing),
        num_rejected=num_rejected,
        split_counts=split_counts,
        processing_seconds=processing_seconds,
    )