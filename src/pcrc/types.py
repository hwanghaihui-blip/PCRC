"""Typed data containers shared across the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PredictionSet:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    score: np.ndarray
    volume: np.ndarray
    tau: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionDecision:
    action: np.ndarray
    risk_value: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutBatch:
    frame: pd.DataFrame


@dataclass(slots=True)
class DatasetBundle:
    train: pd.DataFrame
    calib: pd.DataFrame
    deployment: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
