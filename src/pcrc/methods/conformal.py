"""PCRC closed-loop calibration implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pcrc.methods.base import CalibrationMethod, MethodDiagnostics
from pcrc.models import build_model
from pcrc.solvers.soft_robust import SoftRobustCVaRSolver
from pcrc.types import ActionDecision, PredictionSet


def _weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    q = float(np.clip(quantile, 0.0, 1.0))
    if weights is None:
        return float(np.quantile(arr, q))
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != arr.size:
        raise ValueError("weights and values must have the same shape")
    if np.allclose(w.sum(), 0.0):
        return float(np.quantile(arr, q))
    sorter = np.argsort(arr)
    arr = arr[sorter]
    w = np.clip(w[sorter], 0.0, None)
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, arr))


def _stack_or_empty(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate([np.asarray(chunk, dtype=float).reshape(-1) for chunk in chunks], axis=0)


@dataclass
class PCRCMethod(CalibrationMethod):
    """Performative conformal risk control with closed-loop fixed-point updates."""

    name: str
    solver: SoftRobustCVaRSolver
    model_kind: str = "lgbm_regression"
    nominal_alpha: float = 0.1
    temperature: float = 0.1
    geometry_type: str = "G2"
    seed: int = 0
    device: str = "cuda:1"
    score_normalized: bool = True
    rolling_window: int | None = 1024
    adapt_rate: float = 0.08
    use_decision_geometry: bool = True
    decision_weight: float = 0.40
    utility_weight: float = 0.25
    neutral_action: float | None = None
    safety_margin: float = 0.0
    root_blend: float = 0.5
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(nominal_alpha=self.nominal_alpha, temperature=self.temperature, geometry_type=self.geometry_type)
        self._std_floor = 1e-3
        self.tau = 1.0
        self.alpha_t = float(self.nominal_alpha)
        self.last_action: float | None = None
        self.train_x_: np.ndarray | None = None
        self.train_y_: np.ndarray | None = None
        self.point_model = None
        self.residual_scale_ = 1.0
        self.calibration_scores = np.array([], dtype=float)
        self.history_scores_: list[np.ndarray] = []
        self.history_weights_: list[np.ndarray] = []
        self.history_signals_: list[np.ndarray] = []

    def _spawn_model(self, *, seed_offset: int = 0, quantile: float | None = None, model_kind: str | None = None):
        if self.train_x_ is None:
            raise RuntimeError("fit_base_predictor must be called before spawning task-specific models.")
        return build_model(
            model_kind=model_kind or self.model_kind,
            input_dim=self.train_x_.shape[1],
            seed=self.seed + seed_offset,
            device=self.device,
            quantile=quantile,
        )

    def fit_base_predictor(self, x: np.ndarray, y: np.ndarray) -> None:
        self.train_x_ = np.asarray(x, dtype=float)
        self.train_y_ = np.asarray(y, dtype=float).reshape(-1)
        self.point_model = self._spawn_model(seed_offset=0)
        self.point_model.fit(self.train_x_, self.train_y_)
        center, scale = self._predict_raw(self.point_model, self.train_x_)
        residual = np.abs(self.train_y_ - center)
        self.residual_scale_ = float(max(np.std(residual), np.mean(scale), self._std_floor))

    def _predict_raw(self, model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(x, dtype=float)
        if hasattr(model, "predict_proba"):
            center = np.asarray(model.predict_proba(features), dtype=float).reshape(-1)
            scale = np.sqrt(np.clip(center * (1.0 - center), self._std_floor, None)) + self.residual_scale_
            return center, np.asarray(scale, dtype=float)
        prediction = model.predict(features)
        if isinstance(prediction, tuple):
            center, scale = prediction
            return np.asarray(center, dtype=float).reshape(-1), np.clip(np.asarray(scale, dtype=float).reshape(-1), self._std_floor, None)
        center = np.asarray(prediction, dtype=float).reshape(-1)
        scale = np.full(center.shape[0], max(self.residual_scale_, self._std_floor), dtype=float)
        return center, scale

    def _predict_center_scale(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.point_model is None:
            raise RuntimeError("fit_base_predictor must run before prediction.")
        center, scale = self._predict_raw(self.point_model, x)
        return center, np.clip(scale, self._std_floor, None)

    def _score(self, y: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        residual = np.abs(np.asarray(y, dtype=float).reshape(-1) - np.asarray(center, dtype=float).reshape(-1))
        if self.score_normalized:
            residual = residual / np.clip(np.asarray(scale, dtype=float).reshape(-1), self._std_floor, None)
        return np.asarray(residual, dtype=float)

    def _decision_signal(self, center: np.ndarray) -> np.ndarray:
        magnitude = np.abs(np.tanh(np.asarray(center, dtype=float)))
        base_action = 1.0 if self.last_action is None else float(self.last_action)
        return magnitude * (1.0 + 0.75 * abs(base_action - 1.0))

    def _geometry_multiplier(self, center: np.ndarray) -> np.ndarray:
        if not self.use_decision_geometry:
            return np.ones_like(np.asarray(center, dtype=float))
        signal = self._decision_signal(center)
        geometry = str(self.geometry_type).upper()
        if geometry == "G1":
            return 1.0 + 0.15 * signal
        if geometry == "G2":
            return 1.0 + 0.35 * signal + 0.10 * np.sqrt(np.clip(signal, 0.0, None))
        if geometry == "G3":
            return 1.0 + 0.55 * signal + 0.20 * np.sqrt(np.clip(signal, 0.0, None)) + 0.20 * np.square(signal)
        return 1.0 + 0.20 * signal

    def _make_prediction_set(self, center: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, scale: np.ndarray | None = None) -> PredictionSet:
        center_arr = np.asarray(center, dtype=float).reshape(-1)
        lower_arr = np.asarray(lower, dtype=float).reshape(-1)
        upper_arr = np.asarray(upper, dtype=float).reshape(-1)
        return PredictionSet(
            center=center_arr,
            lower=lower_arr,
            upper=upper_arr,
            score=np.zeros_like(center_arr),
            volume=upper_arr - lower_arr,
            tau=float(self.tau),
            metadata={"scale": np.asarray(scale if scale is not None else np.maximum((upper_arr - lower_arr) / 2.0, self._std_floor), dtype=float)}
            | self.extra_metadata,
        )

    def _interval_from_tau(self, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width = float(self.tau) * np.asarray(scale, dtype=float)
        width = width * self._geometry_multiplier(center)
        if self.safety_margin > 0.0:
            width = width * (1.0 + float(self.safety_margin))
        return np.asarray(center, dtype=float) - width, np.asarray(center, dtype=float) + width

    def _neutral_action(self) -> float:
        if self.neutral_action is not None:
            return float(self.neutral_action)
        grid = np.asarray(self.solver.action_grid, dtype=float)
        return float(np.median(grid)) if grid.size else 0.0

    def _decision_distance(self, action: float) -> float:
        anchor = self._neutral_action()
        previous = anchor if self.last_action is None else float(self.last_action)
        base_distance = abs(float(action) - anchor)
        path_distance = abs(float(action) - previous)
        combined = base_distance + 0.5 * path_distance
        geometry = str(self.geometry_type).upper()
        if geometry == "G1":
            return combined
        if geometry == "G2":
            return combined * (1.0 + np.sqrt(max(combined, 0.0)))
        if geometry == "G3":
            return combined * (1.0 + np.sqrt(max(combined, 0.0)) + max(combined, 0.0))
        return combined

    def _decision_adjusted_costs(self, prediction_set: PredictionSet, candidate_costs: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
        if not candidate_costs:
            return candidate_costs
        if self.decision_weight <= 0.0 and self.utility_weight <= 0.0 and self.safety_margin <= 0.0:
            return candidate_costs
        width = float(np.mean(np.asarray(prediction_set.volume, dtype=float)))
        if width <= 0.0:
            return candidate_costs
        action_means = [float(np.mean(np.asarray(costs, dtype=float))) for costs in candidate_costs.values()]
        within_action_scale = float(np.mean([np.std(np.asarray(costs, dtype=float)) + 1e-6 for costs in candidate_costs.values()]))
        cross_action_scale = float(max(action_means) - min(action_means)) if action_means else 0.0
        cost_scale = within_action_scale + cross_action_scale
        if cost_scale <= 0.0:
            return candidate_costs
        center_signal = float(np.mean(np.abs(np.tanh(np.asarray(prediction_set.center, dtype=float)))))
        adjusted: dict[float, np.ndarray] = {}
        for action, costs in candidate_costs.items():
            distance = self._decision_distance(float(action))
            penalty = width * cost_scale * (
                self.decision_weight * distance * (1.0 + 0.5 * center_signal)
                + self.utility_weight * distance * distance * (1.0 + center_signal)
                + self.safety_margin
            )
            adjusted[float(action)] = np.asarray(costs, dtype=float) + float(penalty)
        return adjusted

    def _push_history(
        self,
        scores: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        signals: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        if score_arr.size == 0:
            return self.calibration_scores.copy(), None, None
        self.history_scores_.append(score_arr)
        if weights is not None:
            self.history_weights_.append(np.asarray(weights, dtype=float).reshape(-1))
        if signals is not None:
            self.history_signals_.append(np.asarray(signals, dtype=float).reshape(-1))
        seed_scores = [self.calibration_scores] if self.calibration_scores.size and not self.history_scores_[:-1] else []
        history_scores = _stack_or_empty(seed_scores + self.history_scores_) if self.history_scores_ else score_arr
        history_weights = _stack_or_empty(self.history_weights_) if self.history_weights_ else None
        history_signals = _stack_or_empty(self.history_signals_) if self.history_signals_ else None
        if seed_scores and history_weights is not None and history_weights.size != history_scores.size:
            prefix = np.ones(seed_scores[0].size, dtype=float)
            history_weights = np.concatenate([prefix, history_weights], axis=0)
        if seed_scores and history_signals is not None and history_signals.size != history_scores.size:
            prefix = np.ones(seed_scores[0].size, dtype=float)
            history_signals = np.concatenate([prefix, history_signals], axis=0)
        if self.rolling_window is not None and history_scores.size > self.rolling_window:
            history_scores = history_scores[-self.rolling_window :]
            if history_weights is not None:
                history_weights = history_weights[-self.rolling_window :]
            if history_signals is not None:
                history_signals = history_signals[-self.rolling_window :]
        self.calibration_scores = history_scores
        return history_scores, history_weights, history_signals

    def fit_uncertainty_module(self, x: np.ndarray, y: np.ndarray) -> None:
        center, scale = self._predict_center_scale(x)
        scores = self._score(np.asarray(y, dtype=float), center, scale)
        signal = 1.0 + self.decision_weight * self._decision_signal(center) + self.utility_weight * np.square(self._decision_signal(center))
        self.calibration_scores = np.asarray(scores, dtype=float)
        self.tau = _weighted_quantile(self.calibration_scores, 1.0 - self.nominal_alpha, weights=signal)

    def predict_distribution_or_set(self, x: np.ndarray) -> PredictionSet:
        center, scale = self._predict_center_scale(x)
        lower, upper = self._interval_from_tau(center, scale)
        return self._make_prediction_set(center, lower, upper, scale=scale)

    def select_action(self, prediction_set: PredictionSet, **kwargs: Any) -> ActionDecision:
        candidate_costs = self._decision_adjusted_costs(prediction_set, kwargs["candidate_costs"])
        decision = self.solver.solve(
            prediction_set.center,
            prediction_set.lower,
            prediction_set.upper,
            candidate_costs,
            previous_action=self.last_action,
        )
        self.last_action = float(decision.action[0])
        return decision

    def offline_fixed_point(self, *args: Any, **kwargs: Any) -> float | None:
        if self.calibration_scores.size == 0:
            return None
        return float(_weighted_quantile(self.calibration_scores, 1.0 - self.nominal_alpha))

    def online_update(self, scores: np.ndarray, covered_post: np.ndarray, **kwargs: Any) -> None:
        score_arr = np.asarray(scores, dtype=float).reshape(-1)
        covered = np.asarray(covered_post, dtype=float).reshape(-1)
        weights = kwargs.get("weights")
        signals = kwargs.get("signals")
        history_scores, history_weights, history_signals = self._push_history(score_arr, weights=weights, signals=signals)
        if history_scores.size == 0:
            return
        target_weights = history_weights
        if target_weights is None and history_signals is not None:
            target_weights = np.clip(history_signals, 1.0, None)
        target_tau = _weighted_quantile(
            history_scores,
            1.0 - self.nominal_alpha,
            weights=target_weights,
        )
        miss = 1.0 - covered
        mean_miss = float(np.average(miss, weights=weights)) if weights is not None else float(miss.mean()) if miss.size else self.nominal_alpha
        rm_tau = max(self._std_floor, self.tau + self.adapt_rate * (mean_miss - self.nominal_alpha))
        self.tau = max(self._std_floor, (1.0 - self.root_blend) * rm_tau + self.root_blend * target_tau)

    def export_diagnostics(self) -> MethodDiagnostics:
        residual = None
        if self.calibration_scores.size:
            residual = float(np.mean(np.abs(self.calibration_scores - self.tau)))
        return MethodDiagnostics(
            tau=float(self.tau),
            temperature=float(self.temperature),
            geometry_type=self.geometry_type,
            calibration_size=int(self.calibration_scores.size),
            fixed_point_residual=residual,
            offline_root=self.offline_fixed_point(),
            extras={"name": self.name, "alpha_t": float(self.alpha_t), "root_blend": float(self.root_blend)} | self.extra_metadata,
        )


def build_method(
    method_name: str,
    action_grid: list[float],
    nominal_alpha: float,
    model_kind: str,
    seed: int,
    device: str,
    beta: float = 0.9,
    temperature: float = 0.1,
    geometry_type: str | None = None,
    method_overrides: dict[str, Any] | None = None,
) -> CalibrationMethod:
    if method_name != "PCRC":
        raise ValueError("This standalone package exposes only the paper's proposed method: PCRC.")
    overrides = dict(method_overrides or {})
    solver = SoftRobustCVaRSolver(
        action_grid=action_grid,
        beta=float(overrides.pop("solver_beta", beta)),
        temperature=float(overrides.pop("solver_temperature", temperature)),
        action_penalty=float(overrides.pop("solver_action_penalty", 0.01)),
    )
    params: dict[str, Any] = dict(
        name="PCRC",
        solver=solver,
        model_kind=model_kind,
        nominal_alpha=nominal_alpha,
        temperature=temperature,
        seed=seed,
        device=device,
        rolling_window=1024,
        adapt_rate=0.08,
        use_decision_geometry=True,
        geometry_type="G2",
        decision_weight=0.40,
        utility_weight=0.25,
        root_blend=0.5,
        extra_metadata={"family": "pcrc"},
    )
    if geometry_type is not None:
        params["geometry_type"] = geometry_type
    params.update(overrides)
    return PCRCMethod(**params)
