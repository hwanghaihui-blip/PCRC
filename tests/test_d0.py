from pcrc.config import ExperimentConfig
from pcrc.experiments.d0 import run_d0_experiment


def test_d0_smoke_runs():
    cfg = ExperimentConfig(
        name="pytest_d0",
        dataset="D0",
        methods=["PCRC"],
        seeds=[7],
        rounds=5,
        calibration_budget=32,
        action_grid=[0.85, 1.0, 1.15],
        params={"regimes": ["S1_stable"], "pretrain_samples": 128, "temperature": 0.1},
    )
    outputs = run_d0_experiment(cfg)
    assert not outputs.rollout_frame.empty
    assert not outputs.summary_frame.empty


def test_d0_accepts_regime_specific_pcrc_overrides():
    cfg = ExperimentConfig(
        name="pytest_d0_regime_overrides",
        dataset="D0",
        methods=["PCRC"],
        seeds=[7],
        rounds=3,
        calibration_budget=16,
        action_grid=[0.85, 1.0, 1.15],
        params={
            "regimes": ["S1_stable", "S2_critical"],
            "pretrain_samples": 64,
            "temperature": 0.1,
            "regime_temperature": {"S2_critical": 0.2},
            "regime_method_overrides": {
                "S1_stable": {"PCRC": {"geometry_type": "G1", "root_blend": 0.2}},
                "S2_critical": {"PCRC": {"geometry_type": "G3", "root_blend": 0.8}},
            },
        },
    )
    outputs = run_d0_experiment(cfg)
    assert set(outputs.rollout_frame["dataset"]) == {"D0:S1_stable", "D0:S2_critical"}
