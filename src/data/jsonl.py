from __future__ import annotations

import json
from collections.abc import Iterable
from types import TracebackType
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


class JsonlWriter:
    """
    Stream JSONL rows to a temporary file and atomically replace the target on success.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_suffix(path.suffix + ".tmp")
        self.count = 0
        self._file = None

    def __enter__(self) -> JsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.temporary_path.open("w", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()

        if exc_type is None:
            self.temporary_path.replace(self.path)
        else:
            self.temporary_path.unlink(missing_ok=True)

    def write(self, row: JsonObject) -> None:
        if self._file is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")

        serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        self._file.write(serialized)
        self._file.write("\n")
        self.count += 1


def read_jsonl(path: Path) -> Iterable[tuple[int, JsonObject]]:
    """
    Read JSONL line by line without loading the whole file into memory.
    """
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, line {line_number}, "
                    f"got {type(value).__name__}"
                )

            yield line_number, value


def write_jsonl(path: Path, rows: Iterable[JsonObject]) -> int:
    """
    Write JSONL atomically and iteratively.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for row in rows:
                serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                file.write(serialized)
                file.write("\n")
                count += 1

        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return count


def write_json(path: Path, value: JsonObject) -> None:
    """
    Write one JSON object atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")

        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
