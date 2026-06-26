"""General utility helpers."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "cuda:1") -> str:
    if requested.startswith("cuda") and torch.cuda.is_available():
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if index < torch.cuda.device_count():
            return requested
    return "cpu"


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(float(np.mean(sample)))
    lower = np.quantile(stats, alpha / 2.0)
    upper = np.quantile(stats, 1.0 - alpha / 2.0)
    return float(arr.mean()), float(lower), float(upper)


def git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        out = "unknown"
    return out


def dump_json(path: str | Path, payload: dict) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_frame(frame: pd.DataFrame, path: str | Path) -> None:
    target = ensure_parent(path)
    if target.suffix == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        frame.to_csv(target, index=False)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def env_summary() -> dict[str, str]:
    return {
        "python": os.popen("python3 --version").read().strip(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_devices": str(torch.cuda.device_count()),
    }
