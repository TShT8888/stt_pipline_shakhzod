from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.jsonl import JsonObject, JsonlWriter, read_jsonl, write_json
from src.data.manifest import resolve_audio_path, row_audio_id, row_duration
from src.data.runtime import torch_runtime_metadata


FEATURE_TYPE = "audio_quality"
FEATURE_VERSION = "1.0"
EPSILON = 1e-12


@dataclass(frozen=True)
class AudioQualityConfig:
    """
    Настройки audio-quality stage.

    Метрики считаются без ASR/ML-модели: stage быстрый, воспроизводимый и подходит
    для labeled, materialized unlabeled VAD clips и будущих manifest-ов.
    """

    silence_threshold_dbfs: float = -50.0
    clipping_threshold: float = 0.999
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    noise_percentile: float = 10.0
    target_min_dbfs: float = -35.0
    target_max_dbfs: float = -8.0


@dataclass(frozen=True)
class AudioQualityOutputs:
    """Сводка результата одного запуска audio-quality stage."""

    features_path: Path
    metadata_path: Path | None
    num_input_rows: int
    num_processed_rows: int
    num_skipped_shard_rows: int
    num_errors: int
    audio_duration: float
    mean_quality_score: float | None
    processing_seconds: float


def amplitude_to_dbfs(value: float) -> float:
    """Переводит линейную амплитуду в dBFS."""
    return 20.0 * math.log10(max(float(value), EPSILON))


def dbfs_to_amplitude(value: float) -> float:
    """Переводит dBFS threshold в линейную амплитуду."""
    return 10.0 ** (value / 20.0)


def load_audio_float32(path: Path) -> tuple[np.ndarray, int]:
    """Загружает аудио как float32 shape=(samples, channels)."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def mono_view(audio: np.ndarray) -> np.ndarray:
    """Возвращает mono-сигнал без лишнего mean для уже mono-аудио."""
    if audio.shape[1] == 1:
        return audio[:, 0]
    return audio.mean(axis=1, dtype=np.float32)


def frame_rms_values(
    mono: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    """
    Считает RMS по фреймам через cumulative sum.

    Так мы не создаем огромную strided-матрицу samples x frames и держим память
    предсказуемой даже для длинных аудио.
    """
    frame_size = max(int(round(sample_rate * frame_ms / 1000.0)), 1)
    hop_size = max(int(round(sample_rate * hop_ms / 1000.0)), 1)
    if mono.size < frame_size:
        return np.asarray([float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))])

    squared = np.square(mono, dtype=np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, mono.size - frame_size + 1, hop_size)
    ends = starts + frame_size
    means = (cumsum[ends] - cumsum[starts]) / frame_size
    return np.sqrt(np.maximum(means, 0.0))


def zero_crossing_rate(mono: np.ndarray, sample_rate: int) -> float:
    """Считает zero-crossing rate в crossings/sec."""
    if mono.size < 2:
        return 0.0

    signs = np.signbit(mono)
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    duration = mono.size / sample_rate if sample_rate > 0 else 0.0
    return crossings / duration if duration > 0 else 0.0


def estimate_quality_score(
    *,
    rms_dbfs: float,
    clipping_ratio: float,
    silence_ratio: float,
    snr_estimate_db: float | None,
    config: AudioQualityConfig,
) -> float:
    """
    Простая эвристика 0..1 для первичной сортировки.

    Это не финальный label "good/bad", а удобный числовой сигнал для будущего selector.
    """
    score = 1.0

    if rms_dbfs < config.target_min_dbfs:
        score -= min((config.target_min_dbfs - rms_dbfs) / 30.0, 0.4)
    if rms_dbfs > config.target_max_dbfs:
        score -= min((rms_dbfs - config.target_max_dbfs) / 20.0, 0.4)

    score -= min(clipping_ratio * 20.0, 0.4)
    score -= min(max(silence_ratio - 0.4, 0.0), 0.4)

    if snr_estimate_db is not None and snr_estimate_db < 10.0:
        score -= min((10.0 - snr_estimate_db) / 30.0, 0.3)

    return round(max(min(score, 1.0), 0.0), 6)


def compute_audio_quality_metrics(
    audio: np.ndarray,
    sample_rate: int,
    config: AudioQualityConfig,
) -> JsonObject:
    """Считает аудио-метрики для одного файла."""
    channels = int(audio.shape[1])
    num_samples = int(audio.shape[0])
    duration = float(num_samples / sample_rate) if sample_rate > 0 else 0.0
    mono = mono_view(audio)
    abs_mono = np.abs(mono)

    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0
    peak = float(np.max(abs_mono)) if mono.size else 0.0
    rms_dbfs = amplitude_to_dbfs(rms)
    peak_dbfs = amplitude_to_dbfs(peak)

    # Silence считаем по sample-level амплитуде. Это простая стабильная метрика,
    # которую потом удобно использовать как фильтр слишком пустых clips.
    silence_threshold = dbfs_to_amplitude(config.silence_threshold_dbfs)
    silence_ratio = float(np.mean(abs_mono <= silence_threshold)) if mono.size else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= config.clipping_threshold)) if audio.size else 0.0

    # DC offset помогает находить криво записанные/сконвертированные аудио.
    channel_means = audio.mean(axis=0, dtype=np.float64) if audio.size else np.asarray([0.0])
    dc_offset_mean_abs = float(np.mean(np.abs(channel_means)))
    dc_offset_max_abs = float(np.max(np.abs(channel_means)))

    # Noise floor оцениваем нижним percentile по frame RMS. Это не студийный SNR,
    # но хороший дешевый сигнал для сортировки и первичной фильтрации.
    frame_rms = frame_rms_values(mono, sample_rate, config.frame_ms, config.hop_ms)
    frame_rms_dbfs = np.asarray([amplitude_to_dbfs(value) for value in frame_rms], dtype=np.float64)
    noise_floor_dbfs = float(np.percentile(frame_rms_dbfs, config.noise_percentile))

    active = abs_mono > silence_threshold
    active_ratio = float(np.mean(active)) if mono.size else 0.0
    active_rms_dbfs: float | None
    snr_estimate_db: float | None
    if np.any(active):
        active_rms = float(np.sqrt(np.mean(np.square(mono[active], dtype=np.float64))))
        active_rms_dbfs = amplitude_to_dbfs(active_rms)
        snr_estimate_db = active_rms_dbfs - noise_floor_dbfs
    else:
        active_rms_dbfs = None
        snr_estimate_db = None

    zcr = zero_crossing_rate(mono, sample_rate)
    quality_score = estimate_quality_score(
        rms_dbfs=rms_dbfs,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        snr_estimate_db=snr_estimate_db,
        config=config,
    )

    return {
        "duration": round(duration, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "num_samples": num_samples,
        "rms_dbfs": round(rms_dbfs, 3),
        "peak_dbfs": round(peak_dbfs, 3),
        "noise_floor_dbfs": round(noise_floor_dbfs, 3),
        "active_rms_dbfs": round(active_rms_dbfs, 3) if active_rms_dbfs is not None else None,
        "snr_estimate_db": round(snr_estimate_db, 3) if snr_estimate_db is not None else None,
        "silence_ratio": round(silence_ratio, 6),
        "active_ratio": round(active_ratio, 6),
        "clipping_ratio": round(clipping_ratio, 8),
        "dc_offset_mean_abs": round(dc_offset_mean_abs, 8),
        "dc_offset_max_abs": round(dc_offset_max_abs, 8),
        "zero_crossing_rate": round(zcr, 3),
        "quality_score": quality_score,
    }


def build_quality_feature_row(
    *,
    source_row: JsonObject,
    input_manifest: Path,
    line_number: int,
    audio_path: Path,
    metrics: JsonObject,
    run_id: str,
) -> JsonObject:
    """Собирает feature-строку для audio-quality."""
    manifest_duration = row_duration(source_row)
    measured_duration = float(metrics["duration"])
    return {
        "audio_id": row_audio_id(source_row),
        "feature_type": FEATURE_TYPE,
        "feature_version": FEATURE_VERSION,
        "status": "ok",
        "source_manifest": str(input_manifest),
        "source_line_number": line_number,
        "audio_path": str(audio_path),
        "dataset": source_row.get("dataset"),
        "language": source_row.get("language"),
        "split": source_row.get("split"),
        "manifest_duration": round(manifest_duration, 3),
        "duration_delta": round(measured_duration - manifest_duration, 3),
        "quality_run_id": run_id,
        **metrics,
    }


def build_quality_error_row(
    *,
    source_row: JsonObject,
    input_manifest: Path,
    line_number: int,
    run_id: str,
    error: Exception,
) -> JsonObject:
    """Собирает error-строку в feature JSONL."""
    return {
        "audio_id": source_row.get("audio_id"),
        "feature_type": FEATURE_TYPE,
        "feature_version": FEATURE_VERSION,
        "status": "error",
        "source_manifest": str(input_manifest),
        "source_line_number": line_number,
        "audio_path": source_row.get("audio_path"),
        "quality_run_id": run_id,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run_audio_quality_manifest(
    *,
    input_manifest: Path,
    output_features: Path,
    output_metadata: Path | None,
    config: AudioQualityConfig,
    run_id: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    fail_fast: bool = False,
) -> AudioQualityOutputs:
    """Считает audio-quality features для audio manifest."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    started_at = time.perf_counter()
    if run_id is None:
        run_id = build_quality_run_id(input_manifest, config, shard_index, num_shards, limit)

    num_input_rows = 0
    num_processed_rows = 0
    num_skipped_shard_rows = 0
    num_errors = 0
    total_audio_duration = 0.0
    quality_score_sum = 0.0
    quality_score_count = 0

    # Каждый feature-stage пишет отдельный JSONL. Мы не модифицируем исходный manifest:
    # финальная сборка train/valid/test потом сделает join по audio_id.
    with JsonlWriter(output_features) as features_writer:
        for line_number, row in read_jsonl(input_manifest):
            is_shard_row = (line_number - 1) % num_shards == shard_index
            if is_shard_row and limit is not None and num_processed_rows >= limit:
                break

            num_input_rows += 1
            if not is_shard_row:
                num_skipped_shard_rows += 1
                continue

            num_processed_rows += 1
            try:
                audio_path = resolve_audio_path(input_manifest, row)
                audio, sample_rate = load_audio_float32(audio_path)
                metrics = compute_audio_quality_metrics(audio, sample_rate, config)
                feature_row = build_quality_feature_row(
                    source_row=row,
                    input_manifest=input_manifest,
                    line_number=line_number,
                    audio_path=audio_path,
                    metrics=metrics,
                    run_id=run_id,
                )
                total_audio_duration += float(metrics["duration"])
                quality_score_sum += float(metrics["quality_score"])
                quality_score_count += 1
            except Exception as exc:
                if fail_fast:
                    raise

                num_errors += 1
                feature_row = build_quality_error_row(
                    source_row=row,
                    input_manifest=input_manifest,
                    line_number=line_number,
                    run_id=run_id,
                    error=exc,
                )

            features_writer.write(feature_row)

    # Metadata описывает весь запуск: удобно сравнивать shards, лимиты и параметры.
    processing_seconds = time.perf_counter() - started_at
    mean_quality_score = (
        quality_score_sum / quality_score_count if quality_score_count > 0 else None
    )
    metadata = {
        "quality_run_id": run_id,
        "feature_type": FEATURE_TYPE,
        "feature_version": FEATURE_VERSION,
        "input_manifest": str(input_manifest),
        "output_features": str(output_features),
        "output_metadata": str(output_metadata) if output_metadata else None,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "limit": limit,
        "num_input_rows": num_input_rows,
        "num_processed_rows": num_processed_rows,
        "num_skipped_shard_rows": num_skipped_shard_rows,
        "num_errors": num_errors,
        "audio_duration": round(total_audio_duration, 3),
        "mean_quality_score": (
            round(mean_quality_score, 6) if mean_quality_score is not None else None
        ),
        "processing_seconds": round(processing_seconds, 3),
        "runtime": runtime_metadata(config),
    }
    if output_metadata is not None:
        write_json(output_metadata, metadata)

    return AudioQualityOutputs(
        features_path=output_features,
        metadata_path=output_metadata,
        num_input_rows=num_input_rows,
        num_processed_rows=num_processed_rows,
        num_skipped_shard_rows=num_skipped_shard_rows,
        num_errors=num_errors,
        audio_duration=total_audio_duration,
        mean_quality_score=mean_quality_score,
        processing_seconds=processing_seconds,
    )


def runtime_metadata(config: AudioQualityConfig) -> JsonObject:
    """Собирает metadata в том же стиле, что VAD/overlap."""
    return torch_runtime_metadata(device="cpu", config_key="audio_quality_config", config=config)


def build_quality_run_id(
    input_manifest: Path,
    config: AudioQualityConfig,
    shard_index: int,
    num_shards: int,
    limit: int | None,
) -> str:
    """Строит стабильный id запуска по manifest, config и shard-параметрам."""
    config_text = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    raw = f"{input_manifest}|{config_text}|{shard_index}|{num_shards}|{limit}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"quality_{digest}"
