from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.jsonl import JsonObject, JsonlWriter, read_jsonl, write_json


SUPPORTED_AUDIO_FORMATS = {"flac", "wav"}


@dataclass(frozen=True)
class MaterializeConfig:
    """
    Настройки физической нарезки VAD-сегментов.

    VAD хранит только offsets внутри исходного файла. Этот этап создает реальные
    audio clips, которые удобно отдавать в pseudo-labeling и fine-tuning.
    """

    output_format: str = "flac"
    subtype: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class MaterializeOutputs:
    manifest_path: Path
    metadata_path: Path | None
    num_input_segments: int
    num_written_segments: int
    num_skipped_existing: int
    num_errors: int
    audio_duration: float
    processing_seconds: float


def load_segments_by_source(segments_path: Path) -> tuple[dict[str, list[JsonObject]], int]:
    """
    Группирует VAD-сегменты по исходному аудио.

    Так мы открываем каждый source WAV один раз и вырезаем из него все куски,
    вместо того чтобы читать один и тот же файл для каждого сегмента отдельно.
    """
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    count = 0
    for _line_number, row in read_jsonl(segments_path):
        source_audio_path = str(row["source_audio_path"])
        grouped[source_audio_path].append(row)
        count += 1
    return grouped, count


def output_audio_path(output_dir: Path, segment_id: str, output_format: str) -> tuple[Path, str]:
    audio_relpath = Path("audio") / f"{segment_id}.{output_format}"
    return output_dir / audio_relpath, audio_relpath.as_posix()


def slice_audio(audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_sample = max(round(start_sec * sample_rate), 0)
    end_sample = min(round(end_sec * sample_rate), audio.shape[0])
    if end_sample <= start_sample:
        return audio[:0]
    return audio[start_sample:end_sample]


def build_materialized_row(
    *,
    segment: JsonObject,
    audio_relpath: str,
    sample_rate: int,
    channels: int,
    actual_duration: float,
) -> JsonObject:
    """Собирает manifest-строку для уже сохраненного audio clip."""
    return {
        "audio_id": segment["segment_id"],
        "audio_path": audio_relpath,
        "duration": round(actual_duration, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "language": segment.get("language"),
        "dataset": segment.get("dataset"),
        "source_audio_id": segment.get("source_audio_id"),
        "source_audio_path": segment.get("source_audio_path"),
        "source_audio_relpath": segment.get("source_audio_relpath"),
        "source_manifest": segment.get("source_manifest"),
        "source_line_number": segment.get("source_line_number"),
        "source_start": segment.get("vad_start"),
        "source_end": segment.get("vad_end"),
        "source_duration": segment.get("source_duration"),
        "vad_duration": segment.get("vad_duration"),
        "vad_run_id": segment.get("vad_run_id"),
        "vad_model": segment.get("vad_model"),
        "vad_threshold": segment.get("vad_threshold"),
    }


def materialize_vad_segments(
    *,
    segments_path: Path,
    output_dir: Path,
    output_manifest: Path | None = None,
    output_metadata: Path | None = None,
    config: MaterializeConfig,
    fail_fast: bool = False,
) -> MaterializeOutputs:
    if config.output_format not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(SUPPORTED_AUDIO_FORMATS)}")

    started_at = time.perf_counter()
    output_manifest = output_manifest or output_dir / "manifest.jsonl"
    grouped_segments, num_input_segments = load_segments_by_source(segments_path)

    num_written_segments = 0
    num_skipped_existing = 0
    num_errors = 0
    audio_duration = 0.0

    with JsonlWriter(output_manifest) as manifest_writer:
        for source_audio_path_text, segments in grouped_segments.items():
            source_audio_path = Path(source_audio_path_text)
            try:
                audio, sample_rate = sf.read(source_audio_path, dtype="float32", always_2d=True)
            except Exception:
                if fail_fast:
                    raise
                num_errors += len(segments)
                continue

            channels = int(audio.shape[1])
            for segment in segments:
                segment_id = str(segment["segment_id"])
                try:
                    clip_path, audio_relpath = output_audio_path(
                        output_dir, segment_id, config.output_format
                    )
                    clip_path.parent.mkdir(parents=True, exist_ok=True)

                    clip = slice_audio(
                        audio,
                        sample_rate,
                        float(segment["vad_start"]),
                        float(segment["vad_end"]),
                    )
                    if clip.shape[0] == 0:
                        raise ValueError("empty audio clip after slicing")

                    if clip_path.exists() and not config.overwrite:
                        num_skipped_existing += 1
                    else:
                        sf.write(
                            clip_path,
                            clip,
                            sample_rate,
                            format=config.output_format.upper(),
                            subtype=config.subtype,
                        )
                        num_written_segments += 1

                    actual_duration = float(clip.shape[0] / sample_rate)
                    audio_duration += actual_duration
                    manifest_writer.write(
                        build_materialized_row(
                            segment=segment,
                            audio_relpath=audio_relpath,
                            sample_rate=sample_rate,
                            channels=channels,
                            actual_duration=actual_duration,
                        )
                    )
                except Exception:
                    if fail_fast:
                        raise
                    num_errors += 1

    processing_seconds = time.perf_counter() - started_at
    outputs = MaterializeOutputs(
        manifest_path=output_manifest,
        metadata_path=output_metadata,
        num_input_segments=num_input_segments,
        num_written_segments=num_written_segments,
        num_skipped_existing=num_skipped_existing,
        num_errors=num_errors,
        audio_duration=audio_duration,
        processing_seconds=processing_seconds,
    )

    if output_metadata is not None:
        write_json(
            output_metadata,
            {
                "segments_path": str(segments_path),
                "output_dir": str(output_dir),
                "output_manifest": str(output_manifest),
                "num_input_segments": outputs.num_input_segments,
                "num_written_segments": outputs.num_written_segments,
                "num_skipped_existing": outputs.num_skipped_existing,
                "num_errors": outputs.num_errors,
                "audio_duration": round(outputs.audio_duration, 3),
                "processing_seconds": round(outputs.processing_seconds, 3),
                "config": {
                    "output_format": config.output_format,
                    "subtype": config.subtype,
                    "overwrite": config.overwrite,
                },
            },
        )

    return outputs
