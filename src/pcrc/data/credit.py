"""UCI credit data download and closed-loop environment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from pcrc.constants import RAW_DATA_DIR


UCI_CREDIT_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def download_credit_dataset() -> Path:
    target_dir = RAW_DATA_DIR / "credit"
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "default_of_credit_card_clients.zip"
    if not archive_path.exists():
        response = requests.get(UCI_CREDIT_URL, timeout=120)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    return archive_path


def load_credit_frame(archive_path: str | Path) -> pd.DataFrame:
    with ZipFile(archive_path) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as handle:
            frame = pd.read_excel(BytesIO(handle.read()), header=1)
    frame = frame.rename(columns={"default payment next month": "default_next_month"})
    frame = frame.drop(columns=["ID"])
    return frame


@dataclass
class CreditResponseKernel:
    lgd: float = 0.75
    interest_margin: float = 0.03
    approval_floor: float = 0.15
    composition_strength: float = 0.25
    near_threshold_band: float = 0.05

    def assign_risk_bins(self, probs: np.ndarray, n_bins: int = 10) -> np.ndarray:
        clipped = np.asarray(probs, dtype=float)
        ranks = pd.qcut(clipped, q=n_bins, labels=False, duplicates="drop")
        arr = np.asarray(ranks, dtype=float)
        return np.nan_to_num(arr, nan=0.0).astype(int)

    def batch_loss(self, approved: pd.DataFrame) -> np.ndarray:
        return self.individual_loss(approved)

    def individual_loss(self, frame: pd.DataFrame) -> np.ndarray:
        default = frame["default_next_month"].to_numpy(dtype=float)
        exposure = frame["LIMIT_BAL"].to_numpy(dtype=float)
        pay_cols = [col for col in frame.columns if str(col).startswith("PAY_")]
        bill_cols = [col for col in frame.columns if str(col).startswith("BILL_AMT")]
        delinquency = np.clip(frame[pay_cols].to_numpy(dtype=float), 0.0, None).mean(axis=1) if pay_cols else np.zeros(len(frame), dtype=float)
        utilization = np.clip(frame[bill_cols].to_numpy(dtype=float), 0.0, None).mean(axis=1) / np.clip(exposure, 1.0, None) if bill_cols else np.zeros(len(frame), dtype=float)
        young_borrower = (frame.get("AGE", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float) < 30.0).astype(float)
        loss_multiplier = 1.0 + 0.08 * delinquency + 0.05 * utilization + 0.03 * young_borrower
        safe_gain = self.interest_margin * (1.0 - default) * np.clip(1.0 - 0.10 * delinquency, 0.4, None)
        return exposure * (self.lgd * loss_multiplier * default - safe_gain)

    def reweight_next_pool(
        self,
        frame: pd.DataFrame,
        threshold: float,
        proba_col: str = "pred_default",
        approval_rate: float | None = None,
        random_state: int | None = 0,
        n_bins: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        probs = frame[proba_col].to_numpy(dtype=float)
        adjusted = frame.copy()
        adjusted = adjusted.drop(columns=["risk_bin", "bin_mid", "selection_weight"], errors="ignore")
        adjusted["risk_bin"] = self.assign_risk_bins(probs, n_bins=n_bins)
        bin_mid = adjusted.groupby("risk_bin")[proba_col].mean().rename("bin_mid").reset_index()
        adjusted = adjusted.merge(bin_mid, on="risk_bin", how="left")
        logits = (float(threshold) - adjusted["bin_mid"].to_numpy(dtype=float)) / max(self.near_threshold_band, 1e-3)
        threshold_distance = np.abs(adjusted[proba_col].to_numpy(dtype=float) - float(threshold)) / max(self.near_threshold_band, 1e-3)
        near_weight = 1.0 / (1.0 + threshold_distance)
        rate = float(approval_rate) if approval_rate is not None else float(np.mean(probs <= float(threshold)))
        approval_pressure = (rate - self.approval_floor) / max(1.0 - self.approval_floor, 1e-3)
        pool_shift = 1.0 / (1.0 + np.exp(-logits))
        adjusted["selection_weight"] = (1.0 - self.composition_strength) + self.composition_strength * pool_shift * (1.0 + 0.35 * near_weight + 0.25 * approval_pressure)
        sampled = adjusted.sample(frac=1.0, replace=True, weights="selection_weight", random_state=random_state).reset_index(drop=True)
        composition = (
            sampled.groupby("risk_bin", as_index=False)
            .agg(share=("risk_bin", "size"))
            .assign(share=lambda df: df["share"] / df["share"].sum(), threshold=float(threshold), approval_rate=float(rate))
        )
        return sampled, composition

    def enforce_approval_floor(self, threshold: float, probs: np.ndarray) -> float:
        candidate = float(threshold)
        approved_rate = float((probs <= candidate).mean())
        if approved_rate >= self.approval_floor:
            return candidate
        grid = np.sort(np.unique(probs))
        for value in grid:
            if float((probs <= value).mean()) >= self.approval_floor:
                return float(value)
        return float(grid[-1])

    def near_threshold_mask(self, probs: np.ndarray, threshold: float) -> np.ndarray:
        return np.abs(np.asarray(probs, dtype=float) - float(threshold)) <= self.near_threshold_band
