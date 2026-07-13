from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from src.data.jsonl import JsonObject, JsonlWriter, read_jsonl, write_json
from src.data.selection import (
    QualityGates,
    build_run_id,
    evaluate_quality_gates,
    load_feature_index,
    selection_score,
)


def default_ssl_gates() -> QualityGates:
    """Сбалансированные пороги под SSL continued pretraining (16 кГц энкодер)."""
    return QualityGates(
        min_duration=3.0,
        max_duration=30.0,
        max_silence_ratio=0.5,
        min_snr_db=8.0,
        max_overlap_ratio=0.15,
    )


@dataclass(frozen=True)
class SslSelectionConfig:
    """
    Отбор materialized unlabeled VAD-клипов под SSL.

    По умолчанию берём всё, что прошло гейты (`target_hours=None`,
    `max_clips_per_source=None`). Diversity-кап и лимит часов включаются явно.
    """

    gates: QualityGates = field(default_factory=default_ssl_gates)
    allowed_languages: tuple[str, ...] | None = None
    max_clips_per_source: int | None = None
    target_hours: float | None = None


@dataclass(frozen=True)
class SslSelectionOutputs:
    selected_path: Path
    rejected_path: Path | None
    metadata_path: Path | None
    num_input_rows: int
    num_selected: int
    num_rejected: int
    selected_hours: float
    processing_seconds: float


def run_ssl_selection(
    *,
    input_manifest: Path,
    quality_features: Sequence[Path],
    overlap_features: Sequence[Path],
    music_features: Sequence[Path],
    output_selected: Path,
    output_rejected: Path | None,
    output_metadata: Path | None,
    config: SslSelectionConfig,
    run_id: str | None = None,
) -> SslSelectionOutputs:
    started_at = time.perf_counter()
    if run_id is None:
        run_id = build_run_id("ssl", input_manifest, config)

    quality_index = load_feature_index(quality_features)
    overlap_index = load_feature_index(overlap_features)
    music_index = load_feature_index(music_features)

    num_input_rows = 0
    num_rejected = 0
    reject_histogram: dict[str, int] = defaultdict(int)
    passing: list[JsonObject] = []

    with ExitStack() as stack:
        rejected_writer = (
            stack.enter_context(JsonlWriter(output_rejected)) if output_rejected else None
        )

        def reject(audio_id: str, row: JsonObject, reasons: list[str]) -> None:
            nonlocal num_rejected
            num_rejected += 1
            for reason in reasons:
                reject_histogram[reason] += 1
            if rejected_writer is not None:
                rejected_writer.write(
                    {
                        "audio_id": audio_id,
                        "reject_reasons": reasons,
                        "duration": row.get("duration"),
                        "language": row.get("language"),
                        "dataset": row.get("dataset"),
                    }
                )

        for _line_number, row in read_jsonl(input_manifest):
            num_input_rows += 1
            audio_id = str(row.get("audio_id"))
            duration = row.get("duration")
            reasons = evaluate_quality_gates(
                duration=float(duration) if duration is not None else None,
                sample_rate=row.get("sample_rate"),
                quality=quality_index.get(audio_id),
                overlap=overlap_index.get(audio_id),
                music=music_index.get(audio_id),
                gates=config.gates,
            )
            language = row.get("language")
            if config.allowed_languages is not None and language not in config.allowed_languages:
                reasons.append("language")

            if reasons:
                reject(audio_id, row, reasons)
                continue

            record = dict(row)
            record["selection_score"] = selection_score(
                quality_index.get(audio_id), overlap_index.get(audio_id)
            )
            record["selection_run_id"] = run_id
            passing.append(record)

        selected = passing

        # Diversity: не даём одной исходной записи забить корпус.
        if config.max_clips_per_source is not None:
            by_source: dict[str, list[JsonObject]] = defaultdict(list)
            for record in passing:
                by_source[str(record.get("source_audio_id"))].append(record)
            selected = []
            for records in by_source.values():
                ordered = sorted(records, key=lambda r: (-r["selection_score"], str(r["audio_id"])))
                selected.extend(ordered[: config.max_clips_per_source])
                for record in ordered[config.max_clips_per_source :]:
                    reject(str(record["audio_id"]), record, ["source_cap"])

        # Бюджет по часам (по умолчанию выключен: берём всё).
        if config.target_hours is not None:
            budget_seconds = config.target_hours * 3600.0
            selected.sort(key=lambda r: (-r["selection_score"], str(r["audio_id"])))
            kept: list[JsonObject] = []
            accumulated = 0.0
            for record in selected:
                clip_seconds = float(record.get("duration") or 0.0)
                if not kept or accumulated + clip_seconds <= budget_seconds:
                    kept.append(record)
                    accumulated += clip_seconds
                else:
                    reject(str(record["audio_id"]), record, ["budget"])
            selected = kept

    selected.sort(key=lambda r: str(r["audio_id"]))
    with JsonlWriter(output_selected) as writer:
        for record in selected:
            writer.write(record)

    selected_hours = sum(float(r.get("duration") or 0.0) for r in selected) / 3600.0
    language_counts: dict[str, int] = defaultdict(int)
    dataset_counts: dict[str, int] = defaultdict(int)
    for record in selected:
        language_counts[str(record.get("language"))] += 1
        dataset_counts[str(record.get("dataset"))] += 1

    processing_seconds = time.perf_counter() - started_at
    if output_metadata is not None:
        write_json(
            output_metadata,
            {
                "selection_run_id": run_id,
                "input_manifest": str(input_manifest),
                "output_selected": str(output_selected),
                "num_input_rows": num_input_rows,
                "num_selected": len(selected),
                "num_rejected": num_rejected,
                "selected_hours": round(selected_hours, 4),
                "reject_histogram": dict(sorted(reject_histogram.items())),
                "selected_by_language": dict(sorted(language_counts.items())),
                "selected_by_dataset": dict(sorted(dataset_counts.items())),
                "processing_seconds": round(processing_seconds, 3),
                "config": _config_to_json(config),
            },
        )

    return SslSelectionOutputs(
        selected_path=output_selected,
        rejected_path=output_rejected,
        metadata_path=output_metadata,
        num_input_rows=num_input_rows,
        num_selected=len(selected),
        num_rejected=num_rejected,
        selected_hours=selected_hours,
        processing_seconds=processing_seconds,
    )


def _config_to_json(config: SslSelectionConfig) -> JsonObject:
    from dataclasses import asdict

    return asdict(config)