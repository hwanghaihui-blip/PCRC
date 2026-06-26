"""Soft-robust CVaR decision solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcrc.types import ActionDecision


def cvar(values: np.ndarray, beta: float = 0.9) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    tail_count = max(1, int(np.ceil((1.0 - beta) * arr.size)))
    return float(arr[-tail_count:].mean())


@dataclass
class SoftRobustCVaRSolver:
    action_grid: list[float]
    beta: float = 0.9
    temperature: float = 0.1
    action_penalty: float = 0.01

    def solve(
        self,
        centers: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        candidate_costs: dict[float, np.ndarray],
        previous_action: float | None = None,
    ) -> ActionDecision:
        best_action = None
        best_risk = float("inf")
        all_risks: dict[float, float] = {}
        for action in self.action_grid:
            costs = np.asarray(candidate_costs[action], dtype=float)
            logits = costs / max(self.temperature, 1e-6)
            soft_weights = np.exp(logits - logits.max())
            soft_weights = soft_weights / soft_weights.sum()
            softened = float(np.sum(soft_weights * costs))
            robust_risk = 0.5 * softened + 0.5 * cvar(costs, beta=self.beta)
            if previous_action is not None:
                robust_risk += self.action_penalty * abs(action - previous_action)
            all_risks[action] = robust_risk
            if robust_risk < best_risk:
                best_action = action
                best_risk = robust_risk
        return ActionDecision(
            action=np.asarray([best_action], dtype=float),
            risk_value=np.asarray([best_risk], dtype=float),
            details={"candidate_risks": all_risks},
        )
