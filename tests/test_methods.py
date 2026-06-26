from __future__ import annotations

import numpy as np
import pytest

from pcrc.methods.conformal import build_method
from pcrc.types import PredictionSet


def test_build_method_applies_pcrc_overrides():
    method = build_method(
        method_name="PCRC",
        action_grid=[0.85, 1.0, 1.15],
        nominal_alpha=0.1,
        model_kind="torch_regression",
        seed=7,
        device="cpu",
        temperature=0.1,
        method_overrides={
            "geometry_type": "G3",
            "root_blend": 0.8,
            "adapt_rate": 0.02,
            "neutral_action": 0.2,
            "solver_beta": 0.95,
            "solver_action_penalty": 0.0,
        },
    )
    assert method.geometry_type == "G3"
    assert method.root_blend == 0.8
    assert method.adapt_rate == 0.02
    assert method.neutral_action == 0.2
    assert method.solver.beta == 0.95
    assert method.solver.action_penalty == 0.0


def test_build_method_respects_explicit_geometry_argument_for_pcrc():
    method = build_method(
        method_name="PCRC",
        action_grid=[0.85, 1.0, 1.15],
        nominal_alpha=0.1,
        model_kind="torch_regression",
        seed=7,
        device="cpu",
        temperature=0.1,
        geometry_type="G3",
    )
    assert method.geometry_type == "G3"


def test_pcrc_online_update_uses_history_quantile_when_windowed():
    method = build_method(
        method_name="PCRC",
        action_grid=[0.85, 1.0, 1.15],
        nominal_alpha=0.1,
        model_kind="torch_regression",
        seed=7,
        device="cpu",
        temperature=0.1,
        method_overrides={"rolling_window": 4, "root_blend": 1.0, "adapt_rate": 0.05},
    )
    method.tau = 1.0
    method.calibration_scores = np.array([1.0, 1.0, 1.0], dtype=float)
    method.online_update(np.array([0.2], dtype=float), np.array([1.0], dtype=float))
    assert method.tau > 0.9
    assert np.isclose(method.tau, np.quantile(np.array([1.0, 1.0, 1.0, 0.2]), 0.9))


def test_geometry_type_changes_prediction_width_when_decision_geometry_enabled():
    x = np.asarray([[0.25, -0.5], [1.0, 0.75]], dtype=float)
    y = np.asarray([0.1, -0.2], dtype=float)
    methods = {}
    for geometry in ["G1", "G2", "G3"]:
        method = build_method(
            method_name="PCRC",
            action_grid=[0.85, 1.0, 1.15],
            nominal_alpha=0.1,
            model_kind="torch_regression",
            seed=7,
            device="cpu",
            temperature=0.1,
            method_overrides={"geometry_type": geometry},
        )
        method.fit_base_predictor(x, y)
        method.tau = 1.0
        methods[geometry] = method.predict_distribution_or_set(x).volume
    assert np.all(methods["G2"] > methods["G1"])
    assert np.all(methods["G3"] > methods["G2"])


def test_pcrc_action_selection_responds_to_prediction_set_width():
    method = build_method(
        method_name="PCRC",
        action_grid=[0.85, 1.0, 1.15],
        nominal_alpha=0.1,
        model_kind="torch_regression",
        seed=7,
        device="cpu",
        temperature=0.1,
        method_overrides={"decision_weight": 1.5, "utility_weight": 0.0, "geometry_type": "G2"},
    )
    candidate_costs = {
        0.85: np.asarray([1.02, 1.03, 1.04], dtype=float),
        1.0: np.asarray([0.99, 1.00, 1.01], dtype=float),
        1.15: np.asarray([0.96, 0.97, 0.98], dtype=float),
    }
    narrow = PredictionSet(
        center=np.asarray([0.5], dtype=float),
        lower=np.asarray([0.45], dtype=float),
        upper=np.asarray([0.55], dtype=float),
        score=np.asarray([0.0], dtype=float),
        volume=np.asarray([0.1], dtype=float),
        tau=1.0,
        metadata={"scale": np.asarray([0.05], dtype=float)},
    )
    wide = PredictionSet(
        center=np.asarray([0.5], dtype=float),
        lower=np.asarray([-0.5], dtype=float),
        upper=np.asarray([1.5], dtype=float),
        score=np.asarray([0.0], dtype=float),
        volume=np.asarray([2.0], dtype=float),
        tau=1.0,
        metadata={"scale": np.asarray([1.0], dtype=float)},
    )
    narrow_action = float(method.select_action(narrow, candidate_costs=candidate_costs).action[0])
    method.last_action = None
    wide_action = float(method.select_action(wide, candidate_costs=candidate_costs).action[0])
    assert narrow_action == 1.15
    assert wide_action == 1.0


def test_zero_weight_action_rule_does_not_change_with_width():
    method = build_method(
        method_name="PCRC",
        action_grid=[0.85, 1.0, 1.15],
        nominal_alpha=0.1,
        model_kind="torch_regression",
        seed=7,
        device="cpu",
        temperature=0.1,
        method_overrides={"decision_weight": 0.0, "utility_weight": 0.0},
    )
    candidate_costs = {
        0.85: np.asarray([1.02, 1.03, 1.04], dtype=float),
        1.0: np.asarray([0.99, 1.00, 1.01], dtype=float),
        1.15: np.asarray([0.96, 0.97, 0.98], dtype=float),
    }
    narrow = PredictionSet(
        center=np.asarray([0.5], dtype=float),
        lower=np.asarray([0.45], dtype=float),
        upper=np.asarray([0.55], dtype=float),
        score=np.asarray([0.0], dtype=float),
        volume=np.asarray([0.1], dtype=float),
        tau=1.0,
        metadata={"scale": np.asarray([0.05], dtype=float)},
    )
    wide = PredictionSet(
        center=np.asarray([0.5], dtype=float),
        lower=np.asarray([-0.5], dtype=float),
        upper=np.asarray([1.5], dtype=float),
        score=np.asarray([0.0], dtype=float),
        volume=np.asarray([2.0], dtype=float),
        tau=1.0,
        metadata={"scale": np.asarray([1.0], dtype=float)},
    )
    narrow_action = float(method.select_action(narrow, candidate_costs=candidate_costs).action[0])
    method.last_action = None
    wide_action = float(method.select_action(wide, candidate_costs=candidate_costs).action[0])
    assert narrow_action == wide_action == 1.15


def test_non_pcrc_methods_are_not_exposed_in_standalone_package():
    with pytest.raises(ValueError, match="only.*PCRC"):
        build_method(
            method_name="not-PCRC",
            action_grid=[0.85, 1.0, 1.15],
            nominal_alpha=0.1,
            model_kind="torch_regression",
            seed=7,
            device="cpu",
            temperature=0.1,
        )
