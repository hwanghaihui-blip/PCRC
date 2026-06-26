"""D0 experiment runner with paper-grade diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcrc.config import ExperimentConfig
from pcrc.constants import REPORTS_DIR, TABLES_APP_DIR
from pcrc.io import artifact_stem, figure_dir_for_run, is_formal_experiment, results_tex_path_for_run
from pcrc.logging_utils import RolloutLogger, RolloutRecord
from pcrc.methods.conformal import build_method
from pcrc.reporting import (
    build_numbered_summary_table,
    export_numbered_table,
    export_table,
    fit_groupwise_linear_trends,
    pairwise_method_tests,
    plot_conditional_mean_curves,
    plot_heatmap,
    plot_method_coverage_panels,
    plot_scatter,
    plot_time_series,
    summarize_metrics,
    write_results_tex,
)
from pcrc.simulators.d0 import D0ClosedLoopSimulator, default_regimes, regime_scan_grid
from pcrc.solvers.soft_robust import cvar
from pcrc.utils import set_global_seed


@dataclass
class D0ExperimentOutputs:
    rollout_frame: pd.DataFrame
    summary_frame: pd.DataFrame
    summary_table: pd.DataFrame
    operator_frame: pd.DataFrame
    trajectory_frame: pd.DataFrame
    phase_frame: pd.DataFrame
    gap_regret_frame: pd.DataFrame
    convergence_summary: pd.DataFrame


def _resolve_regime_method_overrides(config: ExperimentConfig, regime_name: str, method_name: str) -> dict:
    base = dict(config.params.get("method_overrides", {}).get(method_name, {}))
    regime_specific = dict(config.params.get("regime_method_overrides", {}).get(regime_name, {}).get(method_name, {}))
    return base | regime_specific


def _resolve_regime_temperature(config: ExperimentConfig, regime_name: str) -> float:
    regime_temp = config.params.get("regime_temperature", {}).get(regime_name)
    if regime_temp is not None:
        return float(regime_temp)
    return float(config.params.get("temperature", 0.1))


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = sorted(col for col in frame.columns if col.startswith("x"))
    return frame[cols].to_numpy(dtype=float)


def _candidate_costs(simulator: D0ClosedLoopSimulator, x: np.ndarray, action_grid: list[float], n_draws: int) -> dict[float, np.ndarray]:
    context = np.asarray(x, dtype=float).reshape(-1)
    return {
        float(action): simulator.draw_loss_samples(context, float(action), n_draws=n_draws, deterministic=True)
        for action in action_grid
    }


def _static_eval_stream(regime_name: str, seed: int, n_rows: int) -> pd.DataFrame:
    regime = default_regimes()[regime_name]
    stream = D0ClosedLoopSimulator(regime, seed=seed + 517)
    return stream.sample_predeployment(n_rows, anchor_action=1.0)


def _estimate_operator_curve(
    config: ExperimentConfig,
    regime_name: str,
    seed: int,
    tau_grid: np.ndarray,
    n_steps: int,
    initial_taus: list[float],
    model_kind: str,
    mc_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = default_regimes()
    pre_sim = D0ClosedLoopSimulator(regimes[regime_name], seed=seed + 1000, beta=float(config.params.get("beta", 0.9)))
    pre = pre_sim.sample_predeployment(config.params.get("pretrain_samples", 2048), anchor_action=1.0)
    calib = pre.iloc[-config.calibration_budget :].copy()
    train = pre.iloc[: -config.calibration_budget].copy()
    train_x = _feature_matrix(train)
    train_y = train["outcome"].to_numpy(dtype=float)
    calib_x = _feature_matrix(calib)
    calib_y = calib["outcome"].to_numpy(dtype=float)
    operator_rows = []
    trajectory_rows = []
    for tau in tau_grid:
        local_sim = D0ClosedLoopSimulator(regimes[regime_name], seed=seed + int(100 * tau) + 2000, beta=float(config.params.get("beta", 0.9)))
        overrides = _resolve_regime_method_overrides(config, regime_name, "PCRC")
        method = build_method(
            method_name="PCRC",
            action_grid=config.action_grid,
            nominal_alpha=config.nominal_alpha,
            model_kind=model_kind,
            seed=seed,
            device=config.device,
            temperature=_resolve_regime_temperature(config, regime_name),
            beta=float(config.params.get("beta", 0.9)),
            method_overrides=overrides,
        )
        method.fit_base_predictor(train_x, train_y)
        method.fit_uncertainty_module(calib_x, calib_y)
        method.tau = float(tau)
        scores = []
        for round_idx in range(n_steps):
            context_payload = local_sim.sample_context(round_idx)
            x = np.asarray([context_payload[f"x{i}"] for i in range(local_sim.context_dim)], dtype=float).reshape(1, -1)
            prediction = method.predict_distribution_or_set(x)
            decision = method.select_action(
                prediction,
                candidate_costs=_candidate_costs(local_sim, x, config.action_grid, n_draws=mc_draws),
            )
            deployed = local_sim.deploy(round_idx, x.reshape(-1), float(decision.action[0]))
            outcome = float(deployed["outcome"])
            center = float(prediction.center[0])
            scale = float(prediction.metadata["scale"][0])
            score = abs(outcome - center) / max(scale, 1e-6)
            scores.append(score)
        q_post = float(np.quantile(scores, 1.0 - config.nominal_alpha))
        operator_rows.append(
            {
                "dataset": f"D0:{regime_name}",
                "seed": seed,
                "tau_input": float(tau),
                "q_post": q_post,
                "gamma": q_post - float(tau),
                "regime": regime_name,
            }
        )
    for init_tau in initial_taus:
        local_sim = D0ClosedLoopSimulator(regimes[regime_name], seed=seed + int(1000 * init_tau) + 3000, beta=float(config.params.get("beta", 0.9)))
        overrides = _resolve_regime_method_overrides(config, regime_name, "PCRC")
        method = build_method(
            method_name="PCRC",
            action_grid=config.action_grid,
            nominal_alpha=config.nominal_alpha,
            model_kind=model_kind,
            seed=seed,
            device=config.device,
            temperature=_resolve_regime_temperature(config, regime_name),
            beta=float(config.params.get("beta", 0.9)),
            method_overrides=overrides,
        )
        method.fit_base_predictor(train_x, train_y)
        method.fit_uncertainty_module(calib_x, calib_y)
        method.tau = float(init_tau)
        for round_idx in range(n_steps):
            context_payload = local_sim.sample_context(round_idx)
            x = np.asarray([context_payload[f"x{i}"] for i in range(local_sim.context_dim)], dtype=float).reshape(1, -1)
            prediction = method.predict_distribution_or_set(x)
            decision = method.select_action(
                prediction,
                candidate_costs=_candidate_costs(local_sim, x, config.action_grid, n_draws=mc_draws),
            )
            deployed = local_sim.deploy(round_idx, x.reshape(-1), float(decision.action[0]))
            outcome = float(deployed["outcome"])
            center = float(prediction.center[0])
            scale = float(prediction.metadata["scale"][0])
            score = abs(outcome - center) / max(scale, 1e-6)
            covered = float(float(prediction.lower[0]) <= outcome <= float(prediction.upper[0]))
            method.online_update(np.asarray([score]), np.asarray([covered]), signals=np.asarray([1.0 + abs(center)]))
            trajectory_rows.append(
                {
                    "dataset": f"D0:{regime_name}",
                    "seed": seed,
                    "round": round_idx,
                    "tau_init": float(init_tau),
                    "tau_t": float(method.tau),
                    "fp_residual": float(abs(score - method.tau)),
                    "regime": regime_name,
                }
            )
    return pd.DataFrame(operator_rows), pd.DataFrame(trajectory_rows)


def _phase_summary() -> pd.DataFrame:
    frame = regime_scan_grid()
    frame["convergence_speed"] = np.where(frame["stability_margin"] > 0, 1.0 / np.clip(frame["stability_margin"], 1e-3, None), 100.0)
    frame["residual_proxy"] = np.where(frame["stability_margin"] > 0, 1.0 / (1.0 + frame["stability_margin"]), 2.0 + np.abs(frame["stability_margin"]))
    return frame


def _gap_regret_summary(rollout_frame: pd.DataFrame) -> pd.DataFrame:
    gap_frame = rollout_frame.copy()
    gap_frame["coverage_gap"] = gap_frame["covered_pre"] - gap_frame["covered_post"]
    return (
        gap_frame.groupby(["dataset", "method", "seed"], as_index=False)[["coverage_gap", "regret", "risk_value", "fp_residual"]]
        .mean()
        .sort_values(["dataset", "coverage_gap", "regret"])
    )


def _regime_from_dataset(value: str) -> str:
    return str(value).split(":", 1)[-1] if ":" in str(value) else str(value)


def run_d0_experiment(config: ExperimentConfig) -> D0ExperimentOutputs:
    logger = RolloutLogger()
    formal_run = is_formal_experiment(config.name)
    figure_dir = figure_dir_for_run(config.name)
    regimes = default_regimes()
    phase_frame = _phase_summary()
    phase_frame.to_csv("tables/app/d0_regime_scan.csv", index=False)
    operator_frames = []
    trajectory_frames = []
    model_kind = str(config.params.get("model_kind", "lgbm_regression"))
    beta = float(config.params.get("beta", 0.9))
    mc_draws = int(config.params.get("mc_draws", 96))
    for seed in config.seeds:
        set_global_seed(seed)
        for regime_name in config.params.get("regimes", ["S1_stable", "S2_critical", "S3_unstable"]):
            base_simulator = D0ClosedLoopSimulator(regimes[regime_name], seed=seed, beta=beta)
            pre = base_simulator.sample_predeployment(config.params.get("pretrain_samples", 2048), anchor_action=1.0)
            deploy_snapshot = base_simulator.snapshot()
            static_test = _static_eval_stream(regime_name, seed, max(config.rounds, config.params.get("operator_steps", 64)) + 8)
            calib = pre.iloc[-config.calibration_budget :].copy()
            train = pre.iloc[: -config.calibration_budget].copy()
            train_x = _feature_matrix(train)
            train_y = train["outcome"].to_numpy(dtype=float)
            calib_x = _feature_matrix(calib)
            calib_y = calib["outcome"].to_numpy(dtype=float)
            tau_grid = np.linspace(0.50, 2.50, config.params.get("operator_grid_size", 11))
            operator_frame, trajectory_frame = _estimate_operator_curve(
                config=config,
                regime_name=regime_name,
                seed=seed,
                tau_grid=tau_grid,
                n_steps=config.params.get("operator_steps", 64),
                initial_taus=config.params.get("initial_taus", [0.5, 1.0, 1.5, 2.0]),
                model_kind=model_kind,
                mc_draws=mc_draws,
            )
            operator_frames.append(operator_frame)
            trajectory_frames.append(trajectory_frame)
            static_x = _feature_matrix(static_test)
            static_y = static_test["outcome"].to_numpy(dtype=float)
            for method_name in config.methods:
                set_global_seed(seed)
                simulator = base_simulator.fork(deploy_snapshot)
                overrides = _resolve_regime_method_overrides(config, regime_name, method_name)
                method = build_method(
                    method_name=method_name,
                    action_grid=config.action_grid,
                    nominal_alpha=config.nominal_alpha,
                    model_kind=model_kind,
                    seed=seed,
                    device=config.device,
                    beta=beta,
                    temperature=_resolve_regime_temperature(config, regime_name),
                    method_overrides=overrides,
                )
                method.fit_base_predictor(train_x, train_y)
                method.fit_uncertainty_module(calib_x, calib_y)
                for round_idx in range(config.rounds):
                    context_payload = simulator.sample_context(round_idx)
                    x = np.asarray([context_payload[f"x{i}"] for i in range(simulator.context_dim)], dtype=float).reshape(1, -1)
                    prediction = method.predict_distribution_or_set(x)
                    static_prediction = method.predict_distribution_or_set(static_x[round_idx : round_idx + 1])
                    candidate_costs = _candidate_costs(simulator, x, config.action_grid, n_draws=mc_draws)
                    action_decision = method.select_action(prediction, candidate_costs=candidate_costs)
                    deployed_action = float(action_decision.action[0])
                    deployed = simulator.deploy(round_idx, context=x.reshape(-1), action=deployed_action)
                    center = float(prediction.center[0])
                    lower = float(prediction.lower[0])
                    upper = float(prediction.upper[0])
                    outcome = float(deployed["outcome"])
                    scale = float(prediction.metadata["scale"][0])
                    score = abs(outcome - center) / max(scale, 1e-6)
                    covered_pre = float(float(static_prediction.lower[0]) <= float(static_y[round_idx]) <= float(static_prediction.upper[0]))
                    covered_post = float(lower <= outcome <= upper)
                    residual = abs(score - method.tau)
                    losses = candidate_costs[deployed_action]
                    risk_value = float(cvar(losses, beta=beta))
                    logger.log(
                        RolloutRecord(
                            dataset=f"D0:{regime_name}",
                            method=method_name,
                            seed=seed,
                            round=round_idx,
                            context_id=int(deployed["context_id"]),
                            action=deployed_action,
                            outcome=outcome,
                            predicted_center=center,
                            set_lower_or_summary=lower,
                            set_upper_or_summary=upper,
                            set_volume=float(prediction.volume[0]),
                            score=score,
                            tau=float(method.tau),
                            covered_pre=covered_pre,
                            covered_post=covered_post,
                            fp_residual=float(residual),
                            risk_value=risk_value,
                            regret=float(deployed["oracle_regret"]),
                            is_on_policy=1,
                            propensity=1.0 / len(config.action_grid),
                            importance_weight=1.0,
                            ESS=1.0,
                            geometry_type=method.geometry_type,
                            temperature=float(method.temperature),
                            surrogate_misspec_level=0.0,
                        )
                    )
                    method.online_update(np.asarray([score]), np.asarray([covered_post]), signals=np.asarray([1.0 + abs(center)]))
    rollout_frame = logger.to_frame()
    logger.save(config.name)
    summary = summarize_metrics(rollout_frame, ["dataset", "method"])
    summary_table = build_numbered_summary_table(summary, "table2_1", nominal_alpha=config.nominal_alpha)
    operator_frame = pd.concat(operator_frames, ignore_index=True)
    trajectory_frame = pd.concat(trajectory_frames, ignore_index=True)
    gap_regret_frame = _gap_regret_summary(rollout_frame)
    gap_regret_frame["regime"] = gap_regret_frame["dataset"].map(_regime_from_dataset)
    gap_regret_table = fit_groupwise_linear_trends(gap_regret_frame, x="coverage_gap", y="regret", group="regime")
    convergence_summary = summarize_metrics(trajectory_frame, ["regime", "tau_init"], metrics=["tau_t", "fp_residual"]).rename(
        columns={"runtime_or_iteration_count": "round_count"}
    )
    paired_tests = pairwise_method_tests(
        rollout_frame,
        metrics=["covered_post", "risk_value", "regret", "fp_residual"],
        reference_method="PCRC",
        comparison_methods=[method for method in config.methods if method != "PCRC"],
        unit_cols=["dataset", "seed", "round"],
    )
    if formal_run:
        export_numbered_table(summary_table, "table2_1", "coverage_collapse_summary")
        export_numbered_table(convergence_summary, "table2_2", "fixed_point_convergence")
        export_numbered_table(gap_regret_table, "table2_3", "gap_regret_pairs")
    else:
        export_table(summary_table, artifact_stem(config.name, "table2_1_coverage_collapse_summary"), table_dir=TABLES_APP_DIR)
        export_table(convergence_summary, artifact_stem(config.name, "table2_2_fixed_point_convergence"), table_dir=TABLES_APP_DIR)
        export_table(gap_regret_table, artifact_stem(config.name, "table2_3_gap_regret_pairs"), table_dir=TABLES_APP_DIR)
    export_table(paired_tests, f"{config.name}_paired_tests", table_dir=TABLES_APP_DIR)
    export_table(summary, f"{config.name}_raw_summary", table_dir=TABLES_APP_DIR)
    coverage_frame = rollout_frame.melt(
        id_vars=["dataset", "method", "seed", "round"],
        value_vars=["covered_pre", "covered_post"],
        var_name="coverage_type",
        value_name="coverage_value",
    )
    coverage_frame["regime"] = coverage_frame["dataset"].map(_regime_from_dataset)
    plot_method_coverage_panels(
        coverage_frame,
        artifact_stem(config.name, "fig2_1_pre_post_coverage"),
        "Figure 2-1: Pre vs post-decision coverage by round",
        hline=1.0 - config.nominal_alpha,
        figure_dir=figure_dir,
        style="regime",
    )
    plot_time_series(
        rollout_frame,
        "set_volume",
        artifact_stem(config.name, "fig2_2_set_volume"),
        "Figure 2-2: Prediction-set volume by round",
        figure_dir=figure_dir,
    )
    plot_scatter(
        operator_frame,
        "tau_input",
        "q_post",
        artifact_stem(config.name, "fig2_3_operator_curve"),
        "Figure 2-3: Performative quantile operator",
        hue="regime",
        figure_dir=figure_dir,
    )
    plot_time_series(
        trajectory_frame.rename(columns={"tau_t": "covered_post"}),
        "covered_post",
        artifact_stem(config.name, "fig2_4_tau_trajectory"),
        "Figure 2-4: Tau trajectories from different initializations",
        hue="tau_init",
        figure_dir=figure_dir,
    )
    plot_time_series(
        trajectory_frame.rename(columns={"fp_residual": "covered_post"}),
        "covered_post",
        artifact_stem(config.name, "fig2_5_fp_residual"),
        "Figure 2-5: Fixed-point residual by round",
        hue="regime",
        figure_dir=figure_dir,
    )
    plot_heatmap(
        phase_frame,
        "density_scale",
        "performative_intensity",
        "residual_proxy",
        artifact_stem(config.name, "fig2_6_phase_heatmap"),
        "Figure 2-6: Stability heatmap",
        figure_dir=figure_dir,
    )
    representative_phase = trajectory_frame[trajectory_frame["regime"] == "S2_critical"].copy()
    if not representative_phase.empty:
        plot_scatter(
            representative_phase.rename(columns={"tau_t": "coverage_gap", "fp_residual": "regret"}),
            "coverage_gap",
            "regret",
            artifact_stem(config.name, "fig2_7_phase_portrait"),
            "Figure 2-7: Critical-regime phase portrait",
            hue="tau_init",
            figure_dir=figure_dir,
        )
    plot_scatter(
        phase_frame.rename(columns={"performative_intensity": "coverage_gap", "convergence_speed": "regret"}),
        "coverage_gap",
        "regret",
        artifact_stem(config.name, "fig2_8_margin_speed"),
        "Figure 2-8: Convergence speed vs stability margin",
        hue="label",
        figure_dir=figure_dir,
    )
    plot_scatter(
        gap_regret_frame,
        "coverage_gap",
        "regret",
        artifact_stem(config.name, "fig2_9_gap_regret_scatter"),
        "Figure 2-9: Coverage gap to regret",
        figure_dir=figure_dir,
    )
    plot_conditional_mean_curves(
        gap_regret_frame,
        "coverage_gap",
        "regret",
        "regime",
        artifact_stem(config.name, "fig2_10_gap_regret_regime"),
        "Figure 2-10: Coverage gap conditional means by regime",
        figure_dir=figure_dir,
        x_label="Coverage gap",
        y_label="Conditional mean regret",
    )
    write_results_tex("D0 Main Results", summary_table, results_tex_path_for_run(config.name))
    return D0ExperimentOutputs(
        rollout_frame=rollout_frame,
        summary_frame=summary,
        summary_table=summary_table,
        operator_frame=operator_frame,
        trajectory_frame=trajectory_frame,
        phase_frame=phase_frame,
        gap_regret_frame=gap_regret_frame,
        convergence_summary=convergence_summary,
    )
