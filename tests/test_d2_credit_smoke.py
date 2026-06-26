from pcrc.config import ExperimentConfig
from pcrc.experiments.d2_credit import run_d2_credit_experiment


def test_d2_credit_smoke_runs():
    cfg = ExperimentConfig(
        name="pytest_credit_v2",
        dataset="D2",
        methods=["PCRC"],
        seeds=[7],
        rounds=2,
        calibration_budget=128,
        action_grid=[0.15, 0.25, 0.35, 0.45],
        params={
            "model_kind": "credit_classifier",
            "temperature": 0.1,
            "beta": 0.9,
            "approval_floor": 0.15,
            "batch_size": 64,
            "anchor_threshold": 0.3,
            "lgd": 0.75,
            "interest_margin": 0.03,
            "composition_strength": 0.25,
            "near_threshold_band": 0.08,
            "mc_points": 16,
        },
    )
    outputs = run_d2_credit_experiment(cfg)
    assert not outputs.rollout_frame.empty
    assert not outputs.summary_frame.empty
    assert outputs.frame_rows == 30000
