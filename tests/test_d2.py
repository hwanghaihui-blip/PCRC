import numpy as np

from pcrc.experiments.d2_credit import _aggregate_interval, _candidate_cost_draws


def test_aggregate_interval_uses_probability_bands():
    center_loss = np.asarray([12.0, 18.0, 25.0, 33.0])
    lower_loss = np.asarray([10.5, 15.5, 21.0, 28.0])
    upper_loss = np.asarray([13.5, 20.5, 29.0, 37.0])
    loss_scale = np.asarray([1.0, 1.2, 1.4, 1.6])
    mask = np.asarray([True, True, True, False])

    center, lower, upper, scale = _aggregate_interval(
        center_loss=center_loss,
        lower_loss=lower_loss,
        upper_loss=upper_loss,
        loss_scale=loss_scale,
        mask=mask,
        tau=0.25,
    )

    assert np.isfinite(center)
    assert np.isfinite(scale)
    assert lower <= np.mean(lower_loss[mask])
    assert upper >= np.mean(upper_loss[mask])


def test_candidate_cost_draws_expand_with_tau_and_probability_band():
    center_loss = np.asarray([12.0, 18.0, 25.0, 33.0])
    lower_loss = np.asarray([11.7, 17.6, 24.5, 32.4])
    upper_loss = np.asarray([12.3, 18.4, 25.5, 33.6])
    loss_scale = np.asarray([2.0, 2.5, 3.0, 3.5])
    mask = np.asarray([True, True, True, False])

    tight = _candidate_cost_draws(
        center_loss=center_loss,
        lower_loss=lower_loss,
        upper_loss=upper_loss,
        loss_scale=loss_scale,
        mask=mask,
        tau=0.5,
        mc_points=7,
    )
    wide = _candidate_cost_draws(
        center_loss=center_loss,
        lower_loss=lower_loss,
        upper_loss=upper_loss,
        loss_scale=loss_scale,
        mask=mask,
        tau=1.5,
        mc_points=7,
    )

    assert np.isfinite(tight).all()
    assert np.isfinite(wide).all()
    assert np.ptp(wide) > np.ptp(tight)
    assert wide[0] <= tight[0]
    assert wide[-1] > tight[-1]
    assert wide[-1] >= np.mean(upper_loss[mask])
    assert tight[0] <= np.mean(lower_loss[mask])
