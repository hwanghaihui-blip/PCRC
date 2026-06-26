"""Predictive models shared across datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pcrc.utils import pick_device, set_global_seed


_LIGHTGBM_GPU_AVAILABLE: bool | None = None


def lightgbm_gpu_available() -> bool:
    global _LIGHTGBM_GPU_AVAILABLE
    if _LIGHTGBM_GPU_AVAILABLE is not None:
        return _LIGHTGBM_GPU_AVAILABLE
    if not torch.cuda.is_available():
        _LIGHTGBM_GPU_AVAILABLE = False
        return _LIGHTGBM_GPU_AVAILABLE
    try:
        probe_x = np.zeros((32, 4), dtype=float)
        probe_y = np.zeros(32, dtype=float)
        probe = lgb.LGBMRegressor(
            objective="regression",
            device_type="gpu",
            n_estimators=1,
            num_leaves=4,
            min_data_in_leaf=1,
            verbosity=-1,
        )
        probe.fit(probe_x, probe_y)
        _LIGHTGBM_GPU_AVAILABLE = True
    except Exception:
        _LIGHTGBM_GPU_AVAILABLE = False
    return _LIGHTGBM_GPU_AVAILABLE


def _inject_lightgbm_device(params: dict[str, Any], requested_device: str) -> dict[str, Any]:
    enriched = dict(params)
    resolved_device = pick_device(requested_device)
    if resolved_device.startswith("cuda") and lightgbm_gpu_available():
        enriched.setdefault("device_type", "gpu")
        enriched.setdefault("gpu_use_dp", False)
        if ":" in resolved_device:
            enriched.setdefault("gpu_device_id", int(resolved_device.split(":")[1]))
    return enriched


class MeanVarianceMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean = out[:, :1]
        log_var = out[:, 1:]
        return mean, log_var


@dataclass
class TorchRegressor:
    input_dim: int
    hidden_dim: int = 64
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0
    device: str = "cuda:1"

    def __post_init__(self) -> None:
        self.device = pick_device(self.device)
        set_global_seed(self.seed)
        self.model = MeanVarianceMLP(self.input_dim, self.hidden_dim).to(self.device)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchRegressor":
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        y_tensor = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                mean, log_var = self.model(batch_x)
                inv_var = torch.exp(-log_var)
                loss = torch.mean(log_var + (batch_y - mean).pow(2) * inv_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
            mean, log_var = self.model(x_tensor)
            mean_np = mean.squeeze(-1).cpu().numpy()
            std_np = torch.exp(0.5 * log_var).squeeze(-1).cpu().numpy()
        return np.asarray(mean_np, dtype=float), np.asarray(std_np, dtype=float)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict(x)
        return mean


@dataclass
class LightGBMRegressorWrapper:
    params: dict[str, Any]

    def __post_init__(self) -> None:
        merged = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 64,
            "feature_fraction": 0.9,
            "n_jobs": 8,
            "verbosity": -1,
            "random_state": self.params.get("random_state", 0),
        }
        merged.update(self.params)
        self.model = lgb.LGBMRegressor(**merged)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMRegressorWrapper":
        self.model.fit(np.asarray(x), np.asarray(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(x)), dtype=float)


@dataclass
class LightGBMQuantileRegressorWrapper:
    alpha: float
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        merged = {
            "objective": "quantile",
            "alpha": float(self.alpha),
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 64,
            "feature_fraction": 0.9,
            "n_jobs": 8,
            "verbosity": -1,
            "random_state": self.params.get("random_state", 0),
        }
        merged.update(self.params)
        merged["alpha"] = float(self.alpha)
        self.model = lgb.LGBMRegressor(**merged)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMQuantileRegressorWrapper":
        self.model.fit(np.asarray(x), np.asarray(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(x)), dtype=float)


@dataclass
class LightGBMClassifierWrapper:
    params: dict[str, Any]

    def __post_init__(self) -> None:
        merged = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 64,
            "feature_fraction": 0.9,
            "n_jobs": 8,
            "verbosity": -1,
            "random_state": self.params.get("random_state", 0),
        }
        merged.update(self.params)
        self.model = lgb.LGBMClassifier(**merged)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMClassifierWrapper":
        self.model.fit(np.asarray(x), np.asarray(y))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(np.asarray(x))[:, 1], dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x)


def logistic_baseline(seed: int = 0) -> BaseEstimator:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def build_model(
    model_kind: str,
    input_dim: int,
    seed: int,
    device: str,
    *,
    quantile: float | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    cfg = dict(params or {})
    cfg.setdefault("random_state", seed)
    if model_kind in {"lgbm_regression", "credit_classifier"} or quantile is not None:
        cfg = _inject_lightgbm_device(cfg, device)
    if quantile is not None:
        return LightGBMQuantileRegressorWrapper(alpha=float(quantile), params=cfg)
    if model_kind == "torch_regression":
        return TorchRegressor(input_dim=input_dim, seed=seed, device=device)
    if model_kind == "lgbm_regression":
        return LightGBMRegressorWrapper(cfg)
    if model_kind == "credit_classifier":
        return LightGBMClassifierWrapper(cfg)
    if model_kind == "logistic_classifier":
        return logistic_baseline(seed=seed)
    return LightGBMRegressorWrapper(cfg)
