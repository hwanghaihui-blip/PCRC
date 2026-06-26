"""Synthetic D0 closed-loop simulator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcrc.solvers.soft_robust import cvar


@dataclass
class D0Regime:
    name: str
    performative_intensity: float
    density_scale: float
    action_sensitivity: float
    tail_strength: float

    @property
    def stability_margin(self) -> float:
        return float(self.density_scale - self.performative_intensity * self.action_sensitivity)


def default_regimes() -> dict[str, D0Regime]:
    return {
        "S1_stable": D0Regime("S1_stable", performative_intensity=0.35, density_scale=1.30, action_sensitivity=0.60, tail_strength=0.75),
        "S2_critical": D0Regime("S2_critical", performative_intensity=0.55, density_scale=1.00, action_sensitivity=0.70, tail_strength=0.90),
        "S3_unstable": D0Regime("S3_unstable", performative_intensity=0.95, density_scale=0.62, action_sensitivity=1.10, tail_strength=1.30),
    }


class D0ClosedLoopSimulator:
    def __init__(self, regime: D0Regime, context_dim: int = 6, seed: int = 0, beta: float = 0.9) -> None:
        self.regime = regime
        self.context_dim = context_dim
        self.seed = int(seed)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.normal(size=context_dim)

    def snapshot(self) -> dict[str, object]:
        return {
            "state": np.asarray(self.state, dtype=float).copy(),
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.state = np.asarray(snapshot["state"], dtype=float).copy()
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = deepcopy(snapshot["rng_state"])

    def fork(self, snapshot: dict[str, object] | None = None) -> "D0ClosedLoopSimulator":
        clone = D0ClosedLoopSimulator(self.regime, context_dim=self.context_dim, seed=self.seed, beta=self.beta)
        clone.restore(snapshot or self.snapshot())
        return clone

    def _kernel_stats(self, x: np.ndarray, action: float) -> tuple[float, float, float]:
        context = np.asarray(x, dtype=float).reshape(-1)
        action_term = (float(action) - 1.0) * self.regime.action_sensitivity
        response_term = self.regime.performative_intensity * np.tanh(context[0] + 0.5 * context[1]) * action_term
        mean = 3.0 + 1.20 * context[0] - 0.85 * context[1] + 0.60 * context[2] + response_term
        sigma = (0.40 + 0.20 * np.abs(context[3]) + 0.35 * np.abs(action_term)) / max(self.regime.density_scale, 0.1)
        tail = self.regime.tail_strength * (1.0 + 0.25 * np.maximum(context[4], 0.0)) * (1.0 + 0.85 * np.abs(action_term)) / np.sqrt(
            max(self.regime.density_scale, 0.1)
        )
        return float(mean), float(max(sigma, 0.05)), float(max(tail, 0.05))

    def _sample_outcomes(self, x: np.ndarray, action: float, n_draws: int, rng: np.random.Generator | None = None) -> np.ndarray:
        local_rng = rng or self.rng
        mean, sigma, tail = self._kernel_stats(x, action)
        normal = local_rng.normal(loc=0.0, scale=sigma, size=int(n_draws))
        df = max(2.25, 12.0 / (1.0 + tail))
        skew = local_rng.standard_t(df=df, size=int(n_draws)) * 0.10 * tail
        return np.asarray(mean + normal + skew, dtype=float)

    def _loss_from_outcomes(self, x: np.ndarray, action: float, outcomes: np.ndarray) -> np.ndarray:
        context = np.asarray(x, dtype=float).reshape(-1)
        mean, sigma, tail = self._kernel_stats(context, action)
        realized = np.asarray(outcomes, dtype=float)
        revenue = float(action) * realized
        downside = np.clip(2.25 - realized, 0.0, None)
        action_penalty = 0.18 * np.square(float(action) - 1.0) + 0.08 * self.regime.action_sensitivity * np.abs(float(action) - 1.0)
        tail_penalty = 0.18 * tail * np.abs(realized - mean) + 0.12 * sigma * downside
        return -revenue + action_penalty + downside * downside + tail_penalty + 0.04 * np.abs(context[5])

    def _counterfactual_rng(self, x: np.ndarray, action: float, salt: int = 0) -> np.random.Generator:
        context = np.asarray(x, dtype=float).reshape(-1)
        signature = int(np.sum(np.round(np.abs(context) * 1000.0)))
        seed = (self.seed * 100_003 + signature * 97 + int(round(float(action) * 1000)) * 13 + salt) % (2**32 - 1)
        return np.random.default_rng(seed)

    def sample_predeployment(self, n_samples: int, anchor_action: float = 1.0) -> pd.DataFrame:
        rows = []
        local_state = self.state.copy()
        for idx in range(int(n_samples)):
            x = local_state + 0.1 * self.rng.normal(size=self.context_dim)
            outcome = float(self._sample_outcomes(x, anchor_action, 1)[0])
            rows.append(self._pack_row(idx, x, anchor_action, outcome))
            local_state = self._transition(local_state, anchor_action, outcome)
        return pd.DataFrame(rows)

    def sample_context(self, round_idx: int) -> dict:
        x = self.state + 0.10 * self.rng.normal(size=self.context_dim)
        payload = {f"x{i}": float(xi) for i, xi in enumerate(x)}
        payload["context_id"] = int(round_idx)
        return payload

    def anchor_outcome(self, context: np.ndarray, anchor_action: float = 1.0) -> float:
        x = np.asarray(context, dtype=float).reshape(-1)
        rng = self._counterfactual_rng(x, anchor_action, salt=17)
        return float(self._sample_outcomes(x, anchor_action, 1, rng=rng)[0])

    def draw_loss_samples(self, context: np.ndarray, action: float, n_draws: int = 64, deterministic: bool = False) -> np.ndarray:
        x = np.asarray(context, dtype=float).reshape(-1)
        rng = self._counterfactual_rng(x, action, salt=31 if deterministic else 0) if deterministic else self.rng
        outcomes = self._sample_outcomes(x, action, n_draws, rng=rng)
        return np.asarray(self._loss_from_outcomes(x, action, outcomes), dtype=float)

    def oracle_risk(self, x: np.ndarray, action: float, n_draws: int = 256) -> float:
        losses = self.draw_loss_samples(x, action, n_draws=n_draws, deterministic=True)
        return float(cvar(losses, beta=self.beta))

    def oracle_safe_action(self, x: np.ndarray, action_grid: np.ndarray) -> float:
        risks = np.array([self.oracle_risk(x, action) for action in action_grid], dtype=float)
        return float(action_grid[int(np.argmin(risks))])

    def deploy(self, round_idx: int, context: np.ndarray, action: float) -> dict:
        x = np.asarray(context, dtype=float).reshape(-1)
        outcome = float(self._sample_outcomes(x, action, 1)[0])
        mean, sigma, tail = self._kernel_stats(x, action)
        oracle = self.oracle_safe_action(x, np.array([0.85, 0.925, 1.0, 1.075, 1.15], dtype=float))
        regret = self.oracle_risk(x, action) - self.oracle_risk(x, oracle)
        payload = self._pack_row(int(round_idx), x, action, outcome)
        payload.update(
            {
                "mean": mean,
                "sigma": sigma,
                "tail": tail,
                "oracle_action": float(oracle),
                "oracle_regret": float(regret),
                "anchor_outcome": self.anchor_outcome(x, 1.0),
                "realized_loss": float(self._loss_from_outcomes(x, action, np.asarray([outcome], dtype=float))[0]),
            }
        )
        self.state = self._transition(self.state, action, outcome)
        return payload

    def _transition(self, state: np.ndarray, action: float, outcome: float) -> np.ndarray:
        drift = self.regime.performative_intensity * np.array(
            [
                0.55 * (float(action) - 1.0),
                -0.28 * (float(action) - 1.0),
                0.18 * np.tanh(outcome),
                0.15 * np.square(float(action) - 1.0),
                -0.10 * np.tanh(state[0]),
                0.05 * outcome,
            ]
        )
        noise = self.rng.normal(scale=0.04 + 0.05 * self.regime.tail_strength, size=self.context_dim) / np.sqrt(max(self.regime.density_scale, 0.1))
        return 0.84 * np.asarray(state, dtype=float) + drift + noise

    def _pack_row(self, idx: int, x: np.ndarray, action: float, y: float) -> dict:
        payload = {f"x{i}": float(xi) for i, xi in enumerate(np.asarray(x, dtype=float).reshape(-1))}
        payload.update({"context_id": int(idx), "anchor_action": float(action), "outcome": float(y)})
        return payload


def regime_scan_grid() -> pd.DataFrame:
    rows = []
    for intensity in np.linspace(0.10, 1.00, 10):
        for density in np.linspace(0.50, 1.50, 11):
            for sensitivity in np.linspace(0.20, 1.10, 10):
                stability_margin = float(density - intensity * sensitivity)
                if stability_margin > 0.45:
                    label = "convergent"
                elif stability_margin > 0.0:
                    label = "critical"
                else:
                    label = "divergent"
                rows.append(
                    {
                        "performative_intensity": float(intensity),
                        "density_scale": float(density),
                        "action_sensitivity": float(sensitivity),
                        "stability_margin": stability_margin,
                        "label": label,
                    }
                )
    return pd.DataFrame(rows)
