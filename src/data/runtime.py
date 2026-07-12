from __future__ import annotations

import platform
import resource
from dataclasses import asdict
from typing import Any

import torch

from src.data.jsonl import JsonObject


def resolve_torch_device(device: str) -> torch.device:
    """
    Выбирает torch device для stage-скриптов.

    `auto` удобен на сервере: если CUDA доступна, stage запускается на GPU,
    иначе остается CPU. Явный `cuda` падает сразу, если GPU недоступна.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False")

    return resolved


def torch_runtime_metadata(
    *,
    device: str,
    config_key: str,
    config: Any,
) -> JsonObject:
    """Собирает общую runtime metadata для benchmark и аудита stage-запусков."""
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
        config_key: asdict(config),
    }
