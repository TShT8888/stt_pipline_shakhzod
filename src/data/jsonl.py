from __future__ import annotations

import json 
from collections.abc import Iterable, Iterator
from pathlib import Path 
from typing import Any


JsonObject = dict[str, Any]

def read_jsonl(path: Path) -> Iterable[tuple[int, JsonObject]]:
    """
    Читаем JSONL построчно и не загружаем сохраняем в RAM.
    """
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid Json in {path}, line {line_number}: {exc}"
                ) from exc
            
            if not isinstance(value, dict):
                raise ValueError(
                    f"Excepted JSON object in {path} line {line_number}, "
                    f"got {type(value).__name__}"
                )
            
            yield line_number, value 


def write_jsonl(path: Path, rows: Iterable[JsonObject]) -> int:
    """
    Итеративно записываем в JSONL
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    count = 0 
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for row in rows:
                serialized = json.dumps(
                    row,
                    ensure_ascii=False, 
                    separators=(",", ":")
                )
                file.write(serialized)
                file.write("\n")
                count += 1
        
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise 

    return count