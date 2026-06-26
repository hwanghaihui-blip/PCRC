from pcrc.constants import ROLLOUT_COLUMNS
from pcrc.logging_utils import RolloutLogger, RolloutRecord


def test_rollout_logger_columns():
    logger = RolloutLogger()
    logger.log(
        RolloutRecord(
            dataset="D0:S1_stable",
            method="PCRC",
            seed=7,
            round=0,
            context_id=0,
            action=1.0,
            outcome=0.1,
            predicted_center=0.0,
            set_lower_or_summary=-1.0,
            set_upper_or_summary=1.0,
            set_volume=2.0,
            score=0.1,
            tau=1.0,
            covered_pre=1.0,
            covered_post=1.0,
            fp_residual=0.0,
            risk_value=0.1,
            regret=0.0,
            is_on_policy=1,
            propensity=1.0,
            importance_weight=1.0,
            ESS=1.0,
            geometry_type="G2",
            temperature=0.1,
            surrogate_misspec_level=0.0,
        )
    )
    frame = logger.to_frame()
    assert frame.columns.tolist() == ROLLOUT_COLUMNS
    assert len(frame) == 1
