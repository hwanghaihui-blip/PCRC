"""Base interfaces for closed-loop calibration methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pcrc.types import ActionDecision, PredictionSet


@dataclass
class MethodState:
    tau: float
    temperature: float
    geometry_type: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodDiagnostics:
    tau: float
    temperature: float
    geometry_type: str
    calibration_size: int
    fixed_point_residual: float | None = None
    offline_root: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class CalibrationMethod(ABC):
    name: str

    def __init__(self, nominal_alpha: float, temperature: float = 0.1, geometry_type: str = "G1") -> None:
        self.nominal_alpha = nominal_alpha
        self.temperature = temperature
        self.geometry_type = geometry_type
        self.tau = 1.0

    @abstractmethod
    def fit_base_predictor(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_uncertainty_module(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_distribution_or_set(self, x: np.ndarray) -> PredictionSet:
        raise NotImplementedError

    @abstractmethod
    def select_action(self, prediction_set: PredictionSet, **kwargs: Any) -> ActionDecision:
        raise NotImplementedError

    @abstractmethod
    def offline_fixed_point(self, *args: Any, **kwargs: Any) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def online_update(self, scores: np.ndarray, covered_post: np.ndarray, **kwargs: Any) -> None:
        raise NotImplementedError

    def offpolicy_update(
        self,
        scores: np.ndarray,
        covered_post: np.ndarray,
        importance_weight: np.ndarray,
    ) -> None:
        normalized = importance_weight / np.clip(importance_weight.mean(), 1e-6, None)
        self.online_update(scores, covered_post, weights=normalized)

    def fit_base_model(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fit_base_predictor(x, y)

    def calibrate(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fit_uncertainty_module(x, y)

    def predict_set(self, x: np.ndarray) -> PredictionSet:
        return self.predict_distribution_or_set(x)

    def export_state(self) -> MethodState:
        return MethodState(
            tau=float(self.tau),
            temperature=float(self.temperature),
            geometry_type=self.geometry_type,
        )

    def export_diagnostics(self) -> MethodDiagnostics:
        return MethodDiagnostics(
            tau=float(self.tau),
            temperature=float(self.temperature),
            geometry_type=self.geometry_type,
            calibration_size=0,
        )
