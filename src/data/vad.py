from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.data.jsonl import JsonObject, JsonlWriter, read_jsonl
from src.data.jsonl import write_json


DEFAULT_SAMPLE_RATE = 16_000
SILERO_VAD_MODEL = "silero-vad"


@dataclass(frozen=True)
class VadConfig:
    """
    Конфигурация VAD.

    Значения по умолчанию специально чуть консервативные: короткие шумовые
    куски отбрасываются, а пауза в 1 секунду помогает не нарезать одну фразу
    слишком мелко. Для других датасетов эти параметры лучше подбирать benchmark-ом.
    """

    target_sample_rate: int = DEFAULT_SAMPLE_RATE
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 1000
    speech_pad_ms: int = 300
    min_segment_duration: float = 1.5
    max_segment_duration: float = 30.0
    max_threads: int | None = 1


@dataclass(frozen=True)
class VadOutputs:
    """Сводка результата одного запуска VAD."""

    segments_path: Path
    summary_path: Path | None
    metadata_path: Path | None
    num_input_rows: int
    num_processed_rows: int
    num_skipped_shard_rows: int
    num_segments: int
    num_summaries: int
    num_errors: int
    audio_duration: float
    processing_seconds: float
    real_time_factor: float | None


@dataclass(frozen=True)
class VadRuntime:
    """Загруженная модель и функция Silero, чтобы не импортировать их на каждую строку."""

    model: torch.nn.Module
    get_speech_timestamps: Callable[..., list[dict[str, int]]]


def resolve_device(device: str) -> torch.device:
    """
    Выбирает устройство для VAD.

    CPU остается дефолтом в CLI, потому что Silero VAD маленький и на batch size 1
    GPU не всегда быстрее. `auto` оставлен для экспериментов.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False")

    return resolved


def load_vad_runtime(device: torch.device) -> VadRuntime:
    """Загружает Silero VAD один раз на весь процесс."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "silero-vad is required for VAD. Install project dependencies first."
        ) from exc

    model = load_silero_vad()
    model.to(device)
    model.eval()
    return VadRuntime(model=model, get_speech_timestamps=get_speech_timestamps)


def load_audio_mono(path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    """
    Загружает аудио как mono float32 tensor.

    Для mono-файлов не считаем mean, чтобы не создавать лишний массив.
    Если sample rate отличается от целевого, ресемплим через soxr.
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
        sample_rate = target_sample_rate

    return torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32)), sample_rate


def split_long_segment(
    start_sec: float,
    end_sec: float,
    max_duration: float,
) -> list[tuple[float, float]]:
    """
    Fallback-разбиение слишком длинного сегмента.

    Основное ограничение длины делает Silero через `max_speech_duration_s`,
    потому что он старается резать возле пауз. Эта функция нужна только как
    защита, если backend все равно вернул сегмент длиннее лимита.
    """
    if max_duration <= 0:
        raise ValueError("max_duration must be positive")

    if end_sec <= start_sec:
        return []

    if end_sec - start_sec <= max_duration:
        return [(start_sec, end_sec)]

    parts: list[tuple[float, float]] = []
    cursor = start_sec
    while cursor < end_sec:
        part_end = min(cursor + max_duration, end_sec)
        parts.append((cursor, part_end))
        cursor = part_end

    return parts


def resolve_audio_path(input_manifest: Path, audio_path: str) -> Path:
    """Разрешает относительный путь к аудио относительно manifest-файла."""
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return input_manifest.parent / path


def build_segment_row(
    *,
    source_row: JsonObject,
    input_manifest: Path,
    source_audio_path: Path,
    line_number: int,
    vad_index: int,
    part_index: int,
    start_sec: float,
    end_sec: float,
    config: VadConfig,
    vad_run_id: str,
) -> JsonObject:
    """
    Собирает строку segment manifest.

    Важно: здесь мы не режем аудио физически. Строка хранит исходный файл и
    offsets (`vad_start`, `vad_end`), чтобы следующий этап мог материализовать
    реальные clips для pseudo-labeling/fine-tuning.
    """
    source_audio_id = str(source_row["audio_id"])
    segment_id = f"{source_audio_id}_vad_{vad_index:05d}_{part_index:02d}"
    duration = end_sec - start_sec

    return {
        "segment_id": segment_id,
        "source_audio_id": source_audio_id,
        "source_manifest": str(input_manifest),
        "source_line_number": line_number,
        "source_audio_path": str(source_audio_path),
        "source_audio_relpath": source_row.get("audio_path"),
        "dataset": source_row.get("dataset"),
        "language": source_row.get("language"),
        "source_split": source_row.get("split"),
        "source_duration": source_row.get("duration"),
        "source_sample_rate": source_row.get("sample_rate"),
        "source_channels": source_row.get("channels"),
        "source_clips": source_row.get("source_clips"),
        "vad_start": round(start_sec, 3),
        "vad_end": round(end_sec, 3),
        "vad_duration": round(duration, 3),
        "vad_run_id": vad_run_id,
        "vad_model": SILERO_VAD_MODEL,
        "vad_threshold": config.threshold,
    }


def run_vad_for_row(
    *,
    row: JsonObject,
    line_number: int,
    input_manifest: Path,
    runtime: VadRuntime,
    device: torch.device,
    config: VadConfig,
    vad_run_id: str,
) -> tuple[list[JsonObject], JsonObject]:
    """Запускает VAD для одного исходного аудио из manifest."""
    source_audio_id = str(row["audio_id"])
    source_audio_path = resolve_audio_path(input_manifest, str(row["audio_path"]))
    waveform, sample_rate = load_audio_mono(source_audio_path, config.target_sample_rate)
    duration = float(waveform.numel() / sample_rate)
    waveform = waveform.to(device)

    timestamps = runtime.get_speech_timestamps(
        waveform,
        runtime.model,
        sampling_rate=sample_rate,
        threshold=config.threshold,
        min_speech_duration_ms=config.min_speech_duration_ms,
        max_speech_duration_s=config.max_segment_duration,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
        return_seconds=False,
    )

    segments: list[JsonObject] = []
    vad_kept_duration = 0.0
    dropped_short = 0
    fallback_split_long = 0

    for vad_index, timestamp in enumerate(timestamps):
        vad_start = float(timestamp["start"] / sample_rate)
        vad_end = float(timestamp["end"] / sample_rate)

        parts = split_long_segment(vad_start, vad_end, config.max_segment_duration)
        fallback_split_long += max(len(parts) - 1, 0)

        for part_index, (start_sec, end_sec) in enumerate(parts):
            segment_duration = end_sec - start_sec
            if segment_duration < config.min_segment_duration:
                dropped_short += 1
                continue

            # vad_kept_duration включает speech_pad_ms, поэтому это не "чистая речь",
            # а длительность аудио, которую мы оставляем после VAD.
            vad_kept_duration += segment_duration
            segments.append(
                build_segment_row(
                    source_row=row,
                    input_manifest=input_manifest,
                    source_audio_path=source_audio_path,
                    line_number=line_number,
                    vad_index=vad_index,
                    part_index=part_index,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    config=config,
                    vad_run_id=vad_run_id,
                )
            )

    summary: JsonObject = {
        "audio_id": source_audio_id,
        "status": "ok",
        "source_manifest": str(input_manifest),
        "source_line_number": line_number,
        "audio_path": str(source_audio_path),
        "dataset": row.get("dataset"),
        "language": row.get("language"),
        "duration": round(duration, 3),
        "metadata_duration": row.get("duration"),
        "sample_rate": sample_rate,
        "num_vad_segments": len(segments),
        "vad_kept_duration": round(vad_kept_duration, 3),
        "vad_kept_ratio": round(vad_kept_duration / duration, 6) if duration > 0 else 0.0,
        "vad_dropped_duration": round(max(duration - vad_kept_duration, 0.0), 3),
        "dropped_short_segments": dropped_short,
        "fallback_split_long_segments": fallback_split_long,
        "vad_run_id": vad_run_id,
        "vad_model": SILERO_VAD_MODEL,
    }
    return segments, summary


def run_vad_manifest(
    *,
    input_manifest: Path,
    output_segments: Path,
    output_summary: Path | None = None,
    output_metadata: Path | None = None,
    device: str,
    config: VadConfig,
    vad_run_id: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    fail_fast: bool = False,
) -> VadOutputs:
    """
    Запускает VAD по manifest-файлу.

    По умолчанию пишется только `output_segments`, чтобы не плодить много JSON.
    `output_summary` и `output_metadata` включаются явно, когда нужен аудит или benchmark.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    started_at = time.perf_counter()
    if vad_run_id is None:
        vad_run_id = build_vad_run_id(input_manifest, config, shard_index, num_shards)

    torch_device = resolve_device(device)
    if config.max_threads is not None:
        torch.set_num_threads(config.max_threads)

    runtime = load_vad_runtime(torch_device)

    num_input_rows = 0
    num_processed_rows = 0
    num_skipped_shard_rows = 0
    num_errors = 0
    total_audio_duration = 0.0
    num_summaries = 0

    with ExitStack() as output_stack:
        segments_writer = output_stack.enter_context(JsonlWriter(output_segments))
        summary_writer = (
            output_stack.enter_context(JsonlWriter(output_summary))
            if output_summary is not None
            else None
        )
        with torch.inference_mode():
            for line_number, row in read_jsonl(input_manifest):
                num_input_rows += 1

                # Sharding позволяет запускать несколько независимых jobs без общей записи
                # в один JSONL. Это проще и надежнее, чем multiprocessing внутри процесса.
                if (line_number - 1) % num_shards != shard_index:
                    num_skipped_shard_rows += 1
                    continue

                num_processed_rows += 1
                try:
                    segments, summary = run_vad_for_row(
                        row=row,
                        line_number=line_number,
                        input_manifest=input_manifest,
                        runtime=runtime,
                        device=torch_device,
                        config=config,
                        vad_run_id=vad_run_id,
                    )
                    total_audio_duration += float(summary["duration"])
                except Exception as exc:
                    if fail_fast:
                        raise

                    num_errors += 1
                    segments = []
                    summary = {
                        "audio_id": row.get("audio_id"),
                        "status": "error",
                        "source_manifest": str(input_manifest),
                        "source_line_number": line_number,
                        "audio_path": row.get("audio_path"),
                        "vad_run_id": vad_run_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

                for segment in segments:
                    segments_writer.write(segment)
                if summary_writer is not None:
                    summary_writer.write(summary)
                num_summaries += 1

    processing_seconds = time.perf_counter() - started_at
    real_time_factor = (
        processing_seconds / total_audio_duration if total_audio_duration > 0 else None
    )
    metadata = {
        "vad_run_id": vad_run_id,
        "input_manifest": str(input_manifest),
        "output_segments": str(output_segments),
        "output_summary": str(output_summary) if output_summary is not None else None,
        "output_metadata": str(output_metadata) if output_metadata is not None else None,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "num_input_rows": num_input_rows,
        "num_processed_rows": num_processed_rows,
        "num_skipped_shard_rows": num_skipped_shard_rows,
        "num_segments": segments_writer.count,
        "num_summaries": num_summaries,
        "num_errors": num_errors,
        "audio_duration": round(total_audio_duration, 3),
        "processing_seconds": round(processing_seconds, 3),
        "real_time_factor": round(real_time_factor, 6) if real_time_factor is not None else None,
        "device": str(torch_device),
        "runtime": runtime_metadata(str(torch_device), config),
    }
    if output_metadata is not None:
        write_json(output_metadata, metadata)

    return VadOutputs(
        segments_path=output_segments,
        summary_path=output_summary,
        metadata_path=output_metadata,
        num_input_rows=num_input_rows,
        num_processed_rows=num_processed_rows,
        num_skipped_shard_rows=num_skipped_shard_rows,
        num_segments=segments_writer.count,
        num_summaries=num_summaries,
        num_errors=num_errors,
        audio_duration=total_audio_duration,
        processing_seconds=processing_seconds,
        real_time_factor=real_time_factor,
    )


def runtime_metadata(device: str, config: VadConfig) -> JsonObject:
    """Собирает run-level metadata для benchmark и отладки."""
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        peak_rss_mb = peak_rss_mb / (1024 * 1024)
    else:
        peak_rss_mb = peak_rss_mb / 1024

    return {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "peak_rss_mb": round(peak_rss_mb, 3),
        "vad_config": asdict(config),
    }


def build_vad_run_id(
    input_manifest: Path,
    config: VadConfig,
    shard_index: int,
    num_shards: int,
) -> str:
    """Строит стабильный id запуска по manifest, config и shard-параметрам."""
    config_text = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    raw = f"{input_manifest}|{config_text}|{shard_index}|{num_shards}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"vad_{digest}"
