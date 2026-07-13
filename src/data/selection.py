from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.jsonl import JsonObject, read_jsonl


@dataclass(frozen=True)
class QualityGates:
    """
    Общие аудио-гейты отбора. Пороги переопределяются из CLI. Признаки берутся из
    feature-файлов audio_quality/overlap/music по `audio_id`.
    """
    min_duration: float
    max_duration: float

    target_sample_rate: int = 16_000
    require_sample_rate_match: bool = True

    max_clipping_ratio: float = 0.005
    max_silence_ratio: float = 0.30

    min_rms_dbfs: float = -40.0
    max_rms_dbfs: float = -8.0

    min_snr_db: float = 10.0
    max_overlap_ratio: float = 0.10
    max_dc_offset: float = 0.01

    drop_music: bool = True

    require_quality: bool = True
    require_overlap: bool = True
    require_music: bool = True


def load_feature_index(paths: Sequence[Path]) -> dict[str, JsonObject]:
    """Индексирует одну или несколько (шардированных) feature-JSONL по `audio_id`."""
    index: dict[str, JsonObject] = {}
    for path in paths:
        for _line_number, row in read_jsonl(path):
            audio_id = row.get("audio_id")
            if audio_id is not None:
                index[str(audio_id)] = row
    return index


def _status_ok(row: JsonObject | None) -> bool:
    return row is not None and row.get("status") == "ok"


def evaluate_quality_gates(
    *,
    duration: float | None,
    sample_rate: object,
    quality: JsonObject | None,
    overlap: JsonObject | None,
    music: JsonObject | None,
    gates: QualityGates,
) -> list[str]:
    """
    Возвращает список всех проваленных гейтов (пусто = прошло).

    Не short-circuit: собираем все причины, чтобы rejected-аудит и гистограммы
    видели каждую проблему.
    """
    reasons: list[str] = []
    if gates.require_quality and not _status_ok(quality):
        reasons.append("missing_quality" if quality is None else "quality_error")
    if gates.require_overlap and not _status_ok(overlap):
        reasons.append("missing_overlap" if overlap is None else "overlap_error")
    if gates.require_music and not _status_ok(music):
        reasons.append("missing_music" if music is None else "music_error")

    if duration is None:
        reasons.append("missing_duration")
    else:
        if duration < gates.min_duration:
            reasons.append("duration_too_short")
        if duration > gates.max_duration:
            reasons.append("duration_too_long")

    if gates.require_sample_rate_match and sample_rate is not None:
        if int(sample_rate) != gates.target_sample_rate:
            reasons.append("sample_rate_mismatch")

    if _status_ok(quality):
        clipping = quality.get("clipping_ratio")
        if clipping is not None and clipping > gates.max_clipping_ratio:
            reasons.append("clipping")
        silence = quality.get("silence_ratio")
        if silence is not None and silence > gates.max_silence_ratio:
            reasons.append("silence")
        rms = quality.get("rms_dbfs")
        if rms is not None and (rms < gates.min_rms_dbfs or rms > gates.max_rms_dbfs):
            reasons.append("rms_out_of_range")
        snr = quality.get("snr_estimate_db")
        if snr is None:
            reasons.append("snr_missing")
        elif snr < gates.min_snr_db:
            reasons.append("low_snr")
        dc = quality.get("dc_offset_mean_abs")
        if dc is not None and dc > gates.max_dc_offset:
            reasons.append("dc_offset")

    if _status_ok(overlap):
        overlap_ratio = overlap.get("overlap_ratio")
        if overlap_ratio is not None and overlap_ratio > gates.max_overlap_ratio:
            reasons.append("overlap")

    if gates.drop_music and _status_ok(music) and music.get("is_music") is True:
        reasons.append("music")

    return reasons

def selection_score(quality: JsonObject | None, overlap: JsonObject | None) -> float:
    """Сигнал ранжирования/аудита: quality_score − штраф за overlap + бонус за SNR."""
    q = float(quality.get("quality_score", 0.0)) if _status_ok(quality) else 0.0
    ov = float(overlap.get("overlap_ratio", 0.0)) if _status_ok(overlap) else 0.0
    snr = quality.get("snr_estimate_db") if _status_ok(quality) else None
    snr_bonus = min(max((float(snr) if snr is not None else 0.0) / 30.0, 0.0), 1.0) * 0.1
    return round(q - ov * 0.2 + snr_bonus, 6)


def stable_unit_hash(value: str, salt: str = "") -> float:
    """Детерминированный хэш строки в [0, 1) — для воспроизводимого сплита."""
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)



def build_run_id(prefix: str, input_manifest: Path, config: object, extra: str = "") -> str:
    """Стабильный id запуска по manifest, config и доп-параметрам."""
    try:
        config_jsonable: object = asdict(config)  # type: ignore[arg-type]
    except TypeError:
        config_jsonable = str(config)
    config_text = json.dumps(config_jsonable, sort_keys=True, separators=(",", ":"))
    raw = f"{input_manifest}|{config_text}|{extra}".encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:16]}"