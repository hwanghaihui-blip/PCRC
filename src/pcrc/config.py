"""YAML-backed configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pcrc.constants import CONFIGS_DIR, CONFIG_METHOD_ROSTERS, DEFAULT_DEVICE, NOMINAL_ALPHA, SEED_LIST_MAIN


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    dataset: str
    methods: list[str]
    seeds: list[int] = field(default_factory=lambda: list(SEED_LIST_MAIN))
    nominal_alpha: float = NOMINAL_ALPHA
    rounds: int = 200
    calibration_budget: int = 512
    action_grid: list[float] = field(default_factory=list)
    device: str = DEFAULT_DEVICE
    params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def expected_methods(name: str) -> list[str] | None:
        methods = CONFIG_METHOD_ROSTERS.get(name)
        return list(methods) if methods is not None else None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        expected_methods = cls.expected_methods(str(payload["name"]))
        if expected_methods is not None and payload.get("methods") != expected_methods:
            raise ValueError(
                f"Config {payload['name']} must use the canonical method roster {expected_methods}, "
                f"got {payload.get('methods')!r}."
            )
        return cls(**payload)

    def dump(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "name": self.name,
                    "dataset": self.dataset,
                    "methods": self.methods,
                    "seeds": self.seeds,
                    "nominal_alpha": self.nominal_alpha,
                    "rounds": self.rounds,
                    "calibration_budget": self.calibration_budget,
                    "action_grid": self.action_grid,
                    "device": self.device,
                    "params": self.params,
                },
                handle,
                sort_keys=False,
                allow_unicode=True,
            )


def load_config(name: str) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(CONFIGS_DIR / f"{name}.yaml")
