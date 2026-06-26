"""Unified rollout logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pcrc.constants import ROLLOUTS_DIR, ROLLOUT_COLUMNS
from pcrc.utils import ensure_parent


@dataclass(slots=True)
class RolloutRecord:
    dataset: str
    method: str
    seed: int
    round: int
    context_id: int
    action: float
    outcome: float
    predicted_center: float
    set_lower_or_summary: float
    set_upper_or_summary: float
    set_volume: float
    score: float
    tau: float
    covered_pre: float
    covered_post: float
    fp_residual: float
    risk_value: float
    regret: float
    is_on_policy: int
    propensity: float
    importance_weight: float
    ESS: float
    geometry_type: str
    temperature: float
    surrogate_misspec_level: float


class RolloutLogger:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def log(self, record: RolloutRecord) -> None:
        self._records.append(asdict(record))

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self._records)
        if frame.empty:
            return pd.DataFrame(columns=ROLLOUT_COLUMNS)
        missing = [col for col in ROLLOUT_COLUMNS if col not in frame.columns]
        for col in missing:
            frame[col] = pd.NA
        return frame[ROLLOUT_COLUMNS]

    def save(self, stem: str) -> tuple[Path, Path]:
        frame = self.to_frame()
        parquet_path = ensure_parent(ROLLOUTS_DIR / f"{stem}.parquet")
        csv_path = ensure_parent(ROLLOUTS_DIR / f"{stem}.csv")
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(csv_path, index=False)
        return parquet_path, csv_path
