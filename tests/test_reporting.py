from __future__ import annotations

from pathlib import Path

import pandas as pd

from pcrc.contract import table_columns
from pcrc.reporting import apply_coverage_priority, bh_correction, coerce_table_schema, fit_groupwise_linear_trends, manuscript_table_from_ranked, write_results_tex


def test_bh_correction_restores_original_order():
    adjusted = bh_correction([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.04, 0.04]


def test_coerce_table_schema_drops_extra_columns():
    frame = pd.DataFrame(
        [
            {
                "dataset": "D0:S1_stable",
                "method": "PCRC",
                "C_pre": 0.92,
                "C_pre_std": 0.01,
                "C_pre_ci_low": 0.90,
                "C_pre_ci_high": 0.94,
                "C_post": 0.91,
                "C_post_std": 0.01,
                "C_post_ci_low": 0.89,
                "C_post_ci_high": 0.93,
                "gap": 0.01,
                "set_size_or_volume": 1.5,
                "set_size_or_volume_std": 0.1,
                "set_size_or_volume_ci_low": 1.4,
                "set_size_or_volume_ci_high": 1.6,
                "risk(CVaR)": 0.2,
                "risk(CVaR)_std": 0.02,
                "risk(CVaR)_ci_low": 0.18,
                "risk(CVaR)_ci_high": 0.22,
                "regret": 0.05,
                "regret_std": 0.01,
                "regret_ci_low": 0.04,
                "regret_ci_high": 0.06,
                "fp_residual": 0.03,
                "fp_residual_std": 0.01,
                "fp_residual_ci_low": 0.02,
                "fp_residual_ci_high": 0.04,
                "runtime_or_iteration_count": 200,
                "coverage_gate": "pass",
                "coverage_priority_rank": 1,
                "extra_col": "drop-me",
            }
        ]
    )
    coerced = coerce_table_schema(frame, "table2_1")
    assert "extra_col" not in coerced.columns
    assert coerced.columns.tolist() == [
        "dataset",
        "method",
        "C_pre",
        "C_pre_std",
        "C_pre_ci_low",
        "C_pre_ci_high",
        "C_post",
        "C_post_std",
        "C_post_ci_low",
        "C_post_ci_high",
        "gap",
        "set_size_or_volume",
        "set_size_or_volume_std",
        "set_size_or_volume_ci_low",
        "set_size_or_volume_ci_high",
        "risk(CVaR)",
        "risk(CVaR)_std",
        "risk(CVaR)_ci_low",
        "risk(CVaR)_ci_high",
        "regret",
        "regret_std",
        "regret_ci_low",
        "regret_ci_high",
        "fp_residual",
        "fp_residual_std",
        "fp_residual_ci_low",
        "fp_residual_ci_high",
        "runtime_or_iteration_count",
        "coverage_gate",
        "coverage_priority_rank",
    ]


def test_manuscript_table_from_ranked_aliases_machine_summary_columns():
    frame = pd.DataFrame(
        [
            {
                "dataset": "D0:S1_stable",
                "method": "PCRC",
                "covered_pre_mean": 0.92,
                "covered_pre_std": 0.01,
                "covered_pre_ci_low": 0.90,
                "covered_pre_ci_high": 0.94,
                "covered_post_mean": 0.91,
                "covered_post_std": 0.01,
                "covered_post_ci_low": 0.89,
                "covered_post_ci_high": 0.93,
                "gap_mean": 0.01,
                "set_volume_mean": 1.5,
                "set_volume_std": 0.1,
                "set_volume_ci_low": 1.4,
                "set_volume_ci_high": 1.6,
                "risk_value_mean": 0.2,
                "risk_value_std": 0.02,
                "risk_value_ci_low": 0.18,
                "risk_value_ci_high": 0.22,
                "regret_mean": 0.05,
                "regret_std": 0.01,
                "regret_ci_low": 0.04,
                "regret_ci_high": 0.06,
                "fp_residual_mean": 0.03,
                "fp_residual_std": 0.01,
                "fp_residual_ci_low": 0.02,
                "fp_residual_ci_high": 0.04,
                "runtime_or_iteration_count": 200,
                "coverage_gate": "pass",
                "coverage_priority_rank": 1,
            }
        ]
    )
    manuscript = manuscript_table_from_ranked(frame, "table2_1")
    assert manuscript.loc[0, "C_pre"] == 0.92
    assert manuscript.loc[0, "C_post"] == 0.91
    assert manuscript.loc[0, "risk(CVaR)"] == 0.2
    assert manuscript.loc[0, "runtime_or_iteration_count"] == 200


def test_apply_coverage_priority_prefers_pass_then_risk():
    frame = pd.DataFrame(
        [
            {"dataset": "D0:S1_stable", "method": "A", "covered_post_mean": 0.905, "covered_post_ci_low": 0.88, "risk_value_mean": 0.4, "regret_mean": 0.2, "set_volume_mean": 1.0},
            {"dataset": "D0:S1_stable", "method": "B", "covered_post_mean": 0.899, "covered_post_ci_low": 0.87, "risk_value_mean": 0.1, "regret_mean": 0.1, "set_volume_mean": 0.8},
            {"dataset": "D0:S1_stable", "method": "C", "covered_post_mean": 0.86, "covered_post_ci_low": 0.82, "risk_value_mean": 0.01, "regret_mean": 0.01, "set_volume_mean": 0.5},
        ]
    )
    ranked = apply_coverage_priority(frame, nominal_alpha=0.1)
    assert ranked["method"].tolist()[:2] == ["A", "B"]
    assert ranked["coverage_gate"].tolist() == ["pass", "pass", "fail"]


def test_fit_groupwise_linear_trends_returns_regime_level_summary():
    frame = pd.DataFrame(
        [
            {"regime": "S1_stable", "coverage_gap": 0.01, "regret": 0.10},
            {"regime": "S1_stable", "coverage_gap": 0.02, "regret": 0.20},
            {"regime": "S1_stable", "coverage_gap": 0.03, "regret": 0.30},
            {"regime": "S2_critical", "coverage_gap": 0.01, "regret": 0.20},
            {"regime": "S2_critical", "coverage_gap": 0.02, "regret": 0.25},
            {"regime": "S2_critical", "coverage_gap": 0.03, "regret": 0.40},
        ]
    )
    fitted = fit_groupwise_linear_trends(frame, x="coverage_gap", y="regret", group="regime")
    assert fitted.columns.tolist() == ["regime", "slope", "slope_ci_low", "slope_ci_high", "r_squared", "intercept", "pvalue", "n_samples"]
    assert set(fitted["regime"]) == {"S1_stable", "S2_critical"}
    assert (fitted["n_samples"] == 3).all()


def test_write_results_tex_replaces_section_without_truncating_tail(tmp_path: Path):
    path = tmp_path / "results.tex"
    first = pd.DataFrame([{"dataset": "D0", "method": "PCRC", "covered_post_mean": 0.9}])
    second = pd.DataFrame([{"dataset": "D1", "method": "PCRC", "covered_post_mean": 0.91}])
    replacement = pd.DataFrame([{"dataset": "D0", "method": "PCRC", "covered_post_mean": 0.95}])

    write_results_tex("Section A", first, path)
    write_results_tex("Section B", second, path)
    write_results_tex("Section A", replacement, path)

    content = path.read_text(encoding="utf-8")
    assert "% BEGIN Section A" in content
    assert "% BEGIN Section B" in content
    assert "0.950000" in content
    assert "0.910000" in content


def test_specialized_result_tables_keep_core_contract_columns():
    required = {
        "method",
        "C_pre",
        "C_post",
        "gap",
        "set_size_or_volume",
        "risk(CVaR)",
        "regret",
        "fp_residual",
        "runtime_or_iteration_count",
    }
    for table_number in ["table2_2", "table2_3", "table3_2", "table4_2"]:
        assert required.issubset(set(table_columns(table_number)))


def test_coerce_specialized_tables_fills_contract_columns():
    convergence = pd.DataFrame(
        [
            {
                "regime": "S1_stable",
                "tau_init": 0.5,
                "tau_t_mean": 1.2,
                "tau_t_ci_low": 1.1,
                "tau_t_ci_high": 1.3,
                "fp_residual_mean": 0.2,
                "fp_residual_ci_low": 0.1,
                "fp_residual_ci_high": 0.3,
                "round_count": 64,
            }
        ]
    )
    coerced = coerce_table_schema(convergence, "table2_2")
    assert coerced.loc[0, "method"] == "PCRC"
    assert coerced.loc[0, "fp_residual"] == 0.2
    assert coerced.loc[0, "runtime_or_iteration_count"] == 64
    assert coerced.loc[0, "risk(CVaR)"] == "N/A"
