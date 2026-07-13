from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.data.jsonl import JsonObject, JsonlWriter, read_jsonl, write_json
from src.data.manifest import resolve_audio_path, row_audio_id, row_duration
from src.data.runtime import resolve_torch_device, torch_runtime_metadata


MUSIC_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
FEATURE_TYPE = "music_detection"
FEATURE_VERSION = "1.0"


@dataclass(frozen=True)
class MusicDetectionConfig:
    """
    Настройки music-detection stage.

    Stage ожидает обычный audio manifest пайплайна (`audio_id`, `audio_path`,
    `duration`). Такой формат есть у materialized VAD clips и у labeled данных.
    Признак музыки берется из одной верхнеуровневой AudioSet-метки `music_label`.
    """

    model_name: str = MUSIC_MODEL
    target_sample_rate: int = 16_000
    window_seconds: float = 10.0
    music_threshold: float = 0.5
    music_label: str = "Music"
    top_k_labels: int = 5
    batch_size: int = 8
    max_threads: int | None = 1


@dataclass(frozen=True)
class MusicRuntime:
    """Загруженная модель и индекс метки `Music`, чтобы не искать его на каждую строку."""

    model: Any
    feature_extractor: Any
    id2label: dict[int, str]
    label2id: dict[str, int]
    music_index: int


@dataclass(frozen=True)
class MusicDetectionOutputs:
    """Сводка результата одного запуска music-detection stage."""

    features_path: Path
    metadata_path: Path | None
    num_input_rows: int
    num_processed_rows: int
    num_skipped_shard_rows: int
    num_errors: int
    num_music_flagged: int
    audio_duration: float
    processing_seconds: float


def find_label_index(label2id: dict[str, int], name: str) -> int | None:
    """O(1) поиск индекса метки по имени (регистр не важен)."""
    return label2id.get(name.lower())


def load_music_runtime(
    config: MusicDetectionConfig,
    device: torch.device,
    hf_token: str | None = None,
) -> MusicRuntime:
    """Загружает AST AudioSet-классификатор один раз на весь процесс."""
    try:
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for music detection. Install project dependencies first."
        ) from exc

    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name, token=hf_token)
    model = ASTForAudioClassification.from_pretrained(config.model_name, token=hf_token)
    model.to(device)
    model.eval()

    # id2label — с оригинальным регистром (нужен для читаемых top_labels).
    # Ключи label2id приводим к нижнему регистру: поиск O(1) и нечувствителен к
    # регистру (config.music_label может быть "Music" или "music").
    id2label = {int(index): str(name) for index, name in model.config.id2label.items()}
    label2id = {str(name).lower(): int(index) for index, name in model.config.id2label.items()}
    music_index = find_label_index(label2id, config.music_label)
    if music_index is None:
        raise RuntimeError(
            f"Label {config.music_label!r} not found in model {config.model_name!r}"
        )

    return MusicRuntime(
        model=model,
        feature_extractor=feature_extractor,
        id2label=id2label,
        label2id=label2id,
        music_index=music_index,
    )


def load_audio_mono(path: Path, target_sample_rate: int) -> np.ndarray:
    """
    Загружает аудио как mono float32 в целевом sample rate.

    Для mono-файлов не считаем mean, чтобы не плодить лишний массив. Ресемпл
    делаем через soxr, как в VAD-этапе, без torchaudio/ffmpeg.
    """
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        mono = audio[:, 0]
    else:
        mono = audio.mean(axis=1, dtype=np.float32)

    if sample_rate != target_sample_rate:
        try:
            import soxr
        except ImportError as exc:
            raise RuntimeError("soxr is required for fast audio resampling") from exc

        mono = soxr.resample(mono, sample_rate, target_sample_rate).astype(np.float32, copy=False)

    return np.ascontiguousarray(mono, dtype=np.float32)


def frame_into_windows(waveform: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Режет сигнал на неперекрывающиеся окна фиксированной длины.

    Последнее окно дополняется нулями до полной длины. AST ждет фрагмент около 10 с,
    поэтому длинные VAD-клипы прогоняем окнами, а не только по первым 10 секундам.
    """
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")

    length = int(waveform.size)
    num_windows = max(1, math.ceil(length / window_samples))
    padded = np.zeros(num_windows * window_samples, dtype=np.float32)
    padded[:length] = waveform
    return padded.reshape(num_windows, window_samples)


def score_windows(
    runtime: MusicRuntime,
    windows: np.ndarray,
    device: torch.device,
    config: MusicDetectionConfig,
) -> np.ndarray:
    """
    Прогоняет окна через AST батчами и возвращает sigmoid-вероятности.

    Возврат: np.ndarray формы (num_windows, num_labels). Батчинг окон в один forward
    заметно ускоряет инференс и на CPU, и на GPU.
    """
    probabilities: list[np.ndarray] = []
    for start in range(0, len(windows), config.batch_size):
        batch = windows[start : start + config.batch_size]
        inputs = runtime.feature_extractor(
            [window for window in batch],
            sampling_rate=config.target_sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)
        logits = runtime.model(input_values).logits
        probabilities.append(torch.sigmoid(logits).detach().to("cpu").numpy())

    return np.concatenate(probabilities, axis=0)


def top_labels_from_probs(
    mean_probabilities: np.ndarray,
    id2label: dict[int, str],
    top_k: int,
) -> list[JsonObject]:
    """Топ-K меток по средней вероятности между окнами (для аудита, почему помечено)."""
    if top_k <= 0 or mean_probabilities.size == 0:
        return []

    order = np.argsort(mean_probabilities)[::-1][:top_k]
    return [
        {
            "label": id2label.get(int(index), str(int(index))),
            "probability": round(float(mean_probabilities[index]), 6),
        }
        for index in order
    ]


def summarize_window_scores(window_music: np.ndarray, threshold: float) -> JsonObject:
    """Сворачивает per-window вероятности метки `Music` в per-clip сигналы."""
    music_probability = float(window_music.max())
    return {
        "music_probability": round(music_probability, 6),
        "music_mean_probability": round(float(window_music.mean()), 6),
        "music_ratio": round(float(np.mean(window_music >= threshold)), 6),
        "is_music": bool(music_probability >= threshold),
    }


def summarize_music_probs(
    probabilities: np.ndarray,
    runtime: MusicRuntime,
    config: MusicDetectionConfig,
) -> JsonObject:
    """Считает per-clip music-метрики из матрицы вероятностей окон."""
    window_music = probabilities[:, runtime.music_index]
    summary = summarize_window_scores(window_music, config.music_threshold)
    summary["num_windows"] = int(probabilities.shape[0])
    summary["top_labels"] = top_labels_from_probs(
        probabilities.mean(axis=0), runtime.id2label, config.top_k_labels
    )
    return summary


def build_music_feature_row(
    *,
    source_row: JsonObject,
    input_manifest: Path,
    line_number: int,
    audio_path: Path,
    metrics: JsonObject,
    run_id: str,
    config: MusicDetectionConfig,
) -> JsonObject:
    """Собирает feature-строку, которую selector потом join-ит по `audio_id`."""
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
        "duration": round(row_duration(source_row), 3),
        "music_model": config.model_name,
        "music_label": config.music_label,
        "music_threshold": config.music_threshold,
        "music_run_id": run_id,
        **metrics,
    }


def build_music_error_row(
    *,
    source_row: JsonObject,
    input_manifest: Path,
    line_number: int,
    run_id: str,
    error: Exception,
) -> JsonObject:
    """Собирает error-строку в том же feature-файле, чтобы запуск был аудируемым."""
    return {
        "audio_id": source_row.get("audio_id"),
        "feature_type": FEATURE_TYPE,
        "feature_version": FEATURE_VERSION,
        "status": "error",
        "source_manifest": str(input_manifest),
        "source_line_number": line_number,
        "audio_path": source_row.get("audio_path"),
        "music_run_id": run_id,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def detect_music_for_row(
    *,
    row: JsonObject,
    line_number: int,
    input_manifest: Path,
    runtime: MusicRuntime,
    device: torch.device,
    config: MusicDetectionConfig,
    run_id: str,
) -> JsonObject:
    """Считает music-detection фичи для одной manifest-строки."""
    audio_path = resolve_audio_path(input_manifest, row)
    waveform = load_audio_mono(audio_path, config.target_sample_rate)
    window_samples = max(int(round(config.window_seconds * config.target_sample_rate)), 1)
    windows = frame_into_windows(waveform, window_samples)
    probabilities = score_windows(runtime, windows, device, config)
    metrics = summarize_music_probs(probabilities, runtime, config)
    return build_music_feature_row(
        source_row=row,
        input_manifest=input_manifest,
        line_number=line_number,
        audio_path=audio_path,
        metrics=metrics,
        run_id=run_id,
        config=config,
    )


def run_music_detection_manifest(
    *,
    input_manifest: Path,
    output_features: Path,
    output_metadata: Path | None,
    device: str,
    config: MusicDetectionConfig,
    hf_token: str | None = None,
    run_id: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    fail_fast: bool = False,
) -> MusicDetectionOutputs:
    """
    Считает music-detection features для audio manifest.

    Каждый запуск пишет отдельный JSONL через атомарный JsonlWriter. Финальный
    selector join-ит эти строки по `audio_id` и режет клипы с `is_music == true`.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    started_at = time.perf_counter()
    torch_device = resolve_torch_device(device)
    if config.max_threads is not None:
        torch.set_num_threads(config.max_threads)

    if run_id is None:
        run_id = build_music_run_id(input_manifest, config, shard_index, num_shards, limit)

    runtime = load_music_runtime(config, torch_device, hf_token)

    num_input_rows = 0
    num_processed_rows = 0
    num_skipped_shard_rows = 0
    num_errors = 0
    num_music_flagged = 0
    total_audio_duration = 0.0

    with JsonlWriter(output_features) as features_writer, torch.inference_mode():
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
                feature_row = detect_music_for_row(
                    row=row,
                    line_number=line_number,
                    input_manifest=input_manifest,
                    runtime=runtime,
                    device=torch_device,
                    config=config,
                    run_id=run_id,
                )
                total_audio_duration += float(feature_row["duration"])
                if feature_row["is_music"]:
                    num_music_flagged += 1
            except Exception as exc:
                if fail_fast:
                    raise

                num_errors += 1
                feature_row = build_music_error_row(
                    source_row=row,
                    input_manifest=input_manifest,
                    line_number=line_number,
                    run_id=run_id,
                    error=exc,
                )

            features_writer.write(feature_row)

    processing_seconds = time.perf_counter() - started_at
    metadata = {
        "music_run_id": run_id,
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
        "num_music_flagged": num_music_flagged,
        "music_flag_ratio": (
            round(num_music_flagged / num_processed_rows, 6) if num_processed_rows > 0 else None
        ),
        "audio_duration": round(total_audio_duration, 3),
        "processing_seconds": round(processing_seconds, 3),
        "device": str(torch_device),
        "runtime": runtime_metadata(str(torch_device), config),
    }
    if output_metadata is not None:
        write_json(output_metadata, metadata)

    return MusicDetectionOutputs(
        features_path=output_features,
        metadata_path=output_metadata,
        num_input_rows=num_input_rows,
        num_processed_rows=num_processed_rows,
        num_skipped_shard_rows=num_skipped_shard_rows,
        num_errors=num_errors,
        num_music_flagged=num_music_flagged,
        audio_duration=total_audio_duration,
        processing_seconds=processing_seconds,
    )


def runtime_metadata(device: str, config: MusicDetectionConfig) -> JsonObject:
    """Собирает run-level metadata в том же стиле, что VAD/overlap."""
    return torch_runtime_metadata(device=device, config_key="music_config", config=config)


def build_music_run_id(
    input_manifest: Path,
    config: MusicDetectionConfig,
    shard_index: int,
    num_shards: int,
    limit: int | None,
) -> str:
    """Строит стабильный id запуска по manifest, config и shard-параметрам."""
    config_text = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    raw = f"{input_manifest}|{config_text}|{shard_index}|{num_shards}|{limit}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"music_{digest}"