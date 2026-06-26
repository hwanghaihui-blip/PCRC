from pcrc.config import ExperimentConfig
from pcrc.experiments.offpolicy import run_offpolicy_suite


def test_offpolicy_smoke_runs():
    cfg = ExperimentConfig(
        name="pytest_offpolicy_v2",
        dataset="D0",
        methods=["PCRC"],
        seeds=[7],
        rounds=5,
        calibration_budget=32,
        action_grid=[0.85, 1.0, 1.15],
        params={
            "datasets": ["D0"],
            "regime": "S2_critical",
            "overlaps": ["good"],
            "pretrain_samples": 128,
            "temperature": 0.1,
            "model_kind": "lgbm_regression",
            "mc_draws": 16,
        },
    )
    outputs = run_offpolicy_suite(cfg)
    assert not outputs.rollout_frame.empty
    assert not outputs.summary_frame.empty
