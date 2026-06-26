from pcrc.config import ExperimentConfig
from pcrc.experiments.d1_m5 import run_d1_m5_experiment


def test_d1_m5_smoke_runs():
    cfg = ExperimentConfig(
        name="pytest_m5_v2",
        dataset="D1",
        methods=["PCRC"],
        seeds=[7],
        rounds=3,
        calibration_budget=32,
        action_grid=[0.85, 1.0, 1.15],
        params={
            "model_kind": "lgbm_regression",
            "temperature": 0.1,
            "beta": 0.9,
            "monte_carlo_draws": 8,
            "states": ["CA"],
            "categories": ["FOODS"],
            "stores": ["CA_1"],
            "max_series": 20,
        },
    )
    outputs = run_d1_m5_experiment(cfg)
    assert not outputs.rollout_frame.empty
    assert not outputs.summary_frame.empty
    assert outputs.panel_rows > 0
