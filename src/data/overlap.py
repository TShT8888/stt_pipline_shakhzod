from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import soundfile as sf
import torch

from src.data.jsonl import JsonObject, JsonlWriter, read_jsonl, write_json
from src.data.manifest import resolve_audio_path, row_audio_id, row_duration
from src.data.runtime import resolve_torch_device, torch_runtime_metadata


OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
FEATURE_TYPE = "overlap"
FEATURE_VERSION = "1.0"


@dataclass(frozen=True)
class OverlapConfig:
    """
    Настройки overlap-stage.

    Stage ожидает обычный audio manifest из нашей репы: `audio_id`, `audio_path`,
    `duration`. Такой формат есть и у labeled данных, и у materialized VAD clips.
    """

    model_name: str = OVERLAP_MODEL
    min_overlap_duration: float = 0.1
    max_threads: int | None = 1


@dataclass(frozen=True)
class OverlapOutputs:
    """Сводка результата одного запуска overlap-stage."""

    features_path: Path
    metadata_path: Path | None
    num_input_rows: int
    num_processed_rows: int
    num_skipped_shard_rows: int
    num_errors: int
    audio_duration: float
    overlap_duration: float
    processing_seconds: float


def load_audio_for_pyannote(audio_path: Path) -> dict[str, Any]:
    """
    Загружает аудио для pyannote без torchcodec/ffmpeg.

    Pyannote умеет принимать не только путь, но и dict с waveform/sample_rate.
    Так мы используем уже проверенный `soundfile` и избегаем локальных проблем
    с libtorchcodec на macOS. Tensor должен иметь форму (channels, samples).
    """
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))
    return {"waveform": waveform, "sample_rate": sample_rate}


def load_overlap_pipeline(config: OverlapConfig, device: torch.device, hf_token: str | None) -> Any:
    """Загружает pyannote overlap pipeline один раз на весь процесс."""
    # Pyannote импортирует matplotlib. На локальных macOS-сессиях домашний cache
    # иногда недоступен, поэтому направляем cache в системную temp-папку.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "matplotlib"))
    # Pyannote checkpoints - это старые Lightning checkpoints, а не только state_dict.
    # В PyTorch 2.6+ дефолт `weights_only=True` ломает их загрузку. Модель берется
    # из доверенного HuggingFace repo pyannote, поэтому разрешаем обычную загрузку.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    patch_torchaudio_metadata_type()

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is required for overlap detection. Install project dependencies first."
        ) from exc

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    pipeline = Pipeline.from_pretrained(config.model_name, use_auth_token=token)
    pipeline.to(device)
    return pipeline


class AudioMetaDataCompat(NamedTuple):
    """Минимальная совместимость для pyannote.audio с torchaudio без AudioMetaData."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def patch_torchaudio_metadata_type() -> None:
    """
    Совместимость с версиями torchaudio, где `AudioMetaData` больше не экспортируется.

    Pyannote 3.x использует `torchaudio.AudioMetaData` в type annotations при импорте.
    Для работы overlap pipeline нам достаточно вернуть этот тип как NamedTuple-shim.
    """
    try:
        import torchaudio
    except ImportError:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaDataCompat  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]


def iter_overlap_segments(annotation: Any) -> list[JsonObject]:
    """Преобразует pyannote Annotation в компактный список overlap-сегментов."""
    segments: list[JsonObject] = []
    for segment, _, _label in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        duration = max(end - start, 0.0)
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
            }
        )
    return segments


def build_overlap_feature_row(
    *,
    source_row: JsonObject,
    input_manifest: Path,
    line_number: int,
    audio_path: Path,
    overlap_segments: list[JsonObject],
    run_id: str,
    config: OverlapConfig,
) -> JsonObject:
    """Собирает feature-строку, которую потом можно join-ить по `audio_id`."""
    duration = row_duration(source_row)
    overlap_duration = sum(float(segment["duration"]) for segment in overlap_segments)
    overlap_ratio = overlap_duration / duration if duration > 0 else 0.0

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
        "duration": round(duration, 3),
        "overlap_duration": round(overlap_duration, 3),
        "overlap_ratio": round(overlap_ratio, 6),
        "num_overlap_segments": len(overlap_segments),
        "overlap_segments": overlap_segments,
        "overlap_model": config.model_name,
        "overlap_run_id": run_id,
    }


def build_overlap_error_row(
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
        "overlap_run_id": run_id,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run_overlap_manifest(
    *,
    input_manifest: Path,
    output_features: Path,
    output_metadata: Path | None,
    device: str,
    config: OverlapConfig,
    hf_token: str | None = None,
    run_id: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    fail_fast: bool = False,
) -> OverlapOutputs:
    """
    Считает overlap features для audio manifest.

    Каждый запуск пишет отдельный JSONL через атомарный `JsonlWriter`. Для будущих
    quality/audio features создаем отдельные feature-файлы и позже join-им по `audio_id`.
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
        run_id = build_overlap_run_id(input_manifest, config, shard_index, num_shards, limit)

    pipeline = load_overlap_pipeline(config, torch_device, hf_token)

    num_input_rows = 0
    num_processed_rows = 0
    num_skipped_shard_rows = 0
    num_errors = 0
    total_audio_duration = 0.0
    total_overlap_duration = 0.0

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
                audio_path = resolve_audio_path(input_manifest, row)
                duration = row_duration(row)
                annotation = pipeline(load_audio_for_pyannote(audio_path))
                overlap_segments = [
                    segment
                    for segment in iter_overlap_segments(annotation)
                    if float(segment["duration"]) >= config.min_overlap_duration
                ]
                feature_row = build_overlap_feature_row(
                    source_row=row,
                    input_manifest=input_manifest,
                    line_number=line_number,
                    audio_path=audio_path,
                    overlap_segments=overlap_segments,
                    run_id=run_id,
                    config=config,
                )
                total_audio_duration += duration
                total_overlap_duration += float(feature_row["overlap_duration"])
            except Exception as exc:
                if fail_fast:
                    raise

                num_errors += 1
                feature_row = build_overlap_error_row(
                    source_row=row,
                    input_manifest=input_manifest,
                    line_number=line_number,
                    run_id=run_id,
                    error=exc,
                )

            features_writer.write(feature_row)

    processing_seconds = time.perf_counter() - started_at
    metadata = {
        "overlap_run_id": run_id,
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
        "overlap_duration": round(total_overlap_duration, 3),
        "overlap_ratio": (
            round(total_overlap_duration / total_audio_duration, 6)
            if total_audio_duration > 0
            else None
        ),
        "processing_seconds": round(processing_seconds, 3),
        "device": str(torch_device),
        "runtime": runtime_metadata(str(torch_device), config),
    }
    if output_metadata is not None:
        write_json(output_metadata, metadata)

    return OverlapOutputs(
        features_path=output_features,
        metadata_path=output_metadata,
        num_input_rows=num_input_rows,
        num_processed_rows=num_processed_rows,
        num_skipped_shard_rows=num_skipped_shard_rows,
        num_errors=num_errors,
        audio_duration=total_audio_duration,
        overlap_duration=total_overlap_duration,
        processing_seconds=processing_seconds,
    )


def runtime_metadata(device: str, config: OverlapConfig) -> JsonObject:
    """Собирает run-level metadata для overlap-stage."""
    return torch_runtime_metadata(device=device, config_key="overlap_config", config=config)


def build_overlap_run_id(
    input_manifest: Path,
    config: OverlapConfig,
    shard_index: int,
    num_shards: int,
    limit: int | None,
) -> str:
    """Строит стабильный id запуска по manifest, config и shard-параметрам."""
    config_text = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    raw = f"{input_manifest}|{config_text}|{shard_index}|{num_shards}|{limit}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"overlap_{digest}"
