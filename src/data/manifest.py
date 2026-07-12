from __future__ import annotations

from pathlib import Path

from src.data.jsonl import JsonObject


def resolve_audio_path(input_manifest: Path, row: JsonObject) -> Path:
    """Разрешает обязательный `audio_path` относительно manifest-файла."""
    audio_path = row["audio_path"]
    path = Path(str(audio_path))
    if path.is_absolute():
        return path
    return input_manifest.parent / path


def row_audio_id(row: JsonObject) -> str:
    """Возвращает обязательный `audio_id` из manifest-строки."""
    return str(row["audio_id"])


def row_duration(row: JsonObject) -> float:
    """Возвращает обязательную длительность аудио из manifest-строки."""
    return float(row["duration"])
