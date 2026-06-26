"""Off-policy calibration diagnostics on D0 and D2 bandit logs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from pcrc.config import ExperimentConfig
from pcrc.constants import TABLES_APP_DIR
from pcrc.data.credit import CreditResponseKernel, download_credit_dataset, load_credit_frame
from pcrc.experiments.d2_credit import _aggregate_interval, _candidate_cost_draws, _loss_moments, _pseudo_temporal_order, _tail_mean
from pcrc.io import artifact_stem, figure_dir_for_run, is_formal_experiment, results_tex_path_for_run
from pcrc.logging_utils import RolloutLogger, RolloutRecord
from pcrc.methods.conformal import build_method
from pcrc.reporting import (
    apply_coverage_priority,
    export_numbered_table,
    export_table,
    manuscript_table_from_ranked,
    pairwise_method_tests,
    plot_scatter,
    plot_time_series,
    summarize_metrics,
    write_results_tex,
)
from pcrc.simulators.d0 import D0ClosedLoopSimulator, default_regimes
from pcrc.utils import set_global_seed


DEFAULT_OFFPOLICY_METRICS = (
    "covered_pre",
    "covered_post",
    "set_volume",
    "risk_value",
    "regret",
    "fp_residual",
    "ESS",
    "importance_weight",
)


def _resolve_component_method_overrides(config: ExperimentConfig, component: str, method_name: str) -> dict:
    base = dict(config.params.get("method_overrides", {}).get(method_name, {}))
    component_specific = dict(config.params.get("component_method_overrides", {}).get(component, {}).get(method_name, {}))
    return base | component_specific


def _soft_behavior_probs(action_grid: list[float], preferred: float, overlap: str) -> np.ndarray:
    strength = {"good": 0.25, "medium": 0.75, "bad": 1.5}[overlap]
    logits = -strength * np.abs(np.asarray(action_grid, dtype=float) - preferred)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return probs


def _sample_behavior_action(action_grid: list[float], probs: np.ndarray, *, seed: int, component: str, overlap: str, round_idx: int) -> float:
    payload = f"{seed}|{component}|{overlap}|{round_idx}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    local_seed = int.from_bytes(digest, "little") % (2**32 - 1)
    rng = np.random.default_rng(local_seed)
    return float(rng.choice(np.asarray(action_grid, dtype=float), p=np.asarray(probs, dtype=float)))


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = sorted(col for col in frame.columns if col.startswith("x"))
    return frame[cols].to_numpy(dtype=float)


def _prepare_credit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for col in prepared.columns:
        if prepared[col].dtype == "object":
            prepared[col] = prepared[col].astype("category").cat.codes
    return prepared.reset_index(drop=True)


@dataclass
class OffPolicyOutputs:
    rollout_frame: pd.DataFrame
    summary_frame: pd.DataFrame


def _run_d0_component(config: ExperimentConfig, logger: RolloutLogger) -> None:
    regimes = default_regimes()
    action_grid = config.action_grid
    for seed in config.seeds:
        set_global_seed(seed)
        base_simulator = D0ClosedLoopSimulator(regimes[config.params.get("regime", "S2_critical")], seed=seed)
        pre = base_simulator.sample_predeployment(config.params.get("pretrain_samples", 1024))
        deploy_snapshot = base_simulator.snapshot()
        train = pre.iloc[: -config.calibration_budget]
        calib = pre.iloc[-config.calibration_budget :]
        x_cols = sorted(col for col in pre.columns if col.startswith("x"))
        train_x = train[x_cols].to_numpy(dtype=float)
        train_y = train["outcome"].to_numpy(dtype=float)
        calib_x = calib[x_cols].to_numpy(dtype=float)
        calib_y = calib["outcome"].to_numpy(dtype=float)
        for overlap in config.params.get("overlaps", ["good", "medium", "bad"]):
            for method_name in config.methods:
                set_global_seed(seed)
                simulator = base_simulator.fork(deploy_snapshot)
                method = build_method(
                    method_name,
                    action_grid,
                    config.nominal_alpha,
                    str(config.params.get("model_kind", "lgbm_regression")),
                    seed,
                    config.device,
                    beta=float(config.params.get("beta", 0.9)),
                    temperature=float(config.params.get("temperature", 0.1)),
                    method_overrides=_resolve_component_method_overrides(config, "D0", method_name),
                )
                method.fit_base_predictor(train_x, train_y)
                method.fit_uncertainty_module(calib_x, calib_y)
                history_scores: list[float] = []
                history_cov: list[float] = []
                history_weights: list[float] = []
                for round_idx in range(config.rounds):
                    context_payload = simulator.sample_context(round_idx)
                    x = np.asarray([context_payload[col] for col in x_cols], dtype=float).reshape(1, -1)
                    prediction = method.predict_distribution_or_set(x)
                    candidate_costs = {}
                    for action in action_grid:
                        candidate_costs[action] = simulator.draw_loss_samples(
                            x.reshape(-1),
                            action,
                            n_draws=int(config.params.get("mc_draws", 64)),
                            deterministic=True,
                        )
                    target_decision = method.select_action(prediction, candidate_costs=candidate_costs)
                    target_action = float(target_decision.action[0])
                    behavior_probs = _soft_behavior_probs(action_grid, preferred=1.0, overlap=overlap)
                    behavior_action = _sample_behavior_action(
                        action_grid,
                        behavior_probs,
                        seed=seed,
                        component="D0",
                        overlap=overlap,
                        round_idx=round_idx,
                    )
                    propensity = float(behavior_probs[action_grid.index(behavior_action)])
                    target_propensity = float(_soft_behavior_probs(action_grid, preferred=target_action, overlap="good")[action_grid.index(behavior_action)])
                    weight = target_propensity / max(propensity, 1e-6)
                    if method_name == "PCRC":
                        weight = 1.0
                    deployed = simulator.deploy(round_idx, x.reshape(-1), behavior_action)
                    center = float(prediction.center[0])
                    lower = float(prediction.lower[0])
                    upper = float(prediction.upper[0])
                    outcome = float(deployed["outcome"])
                    score = abs(outcome - center) / max(float(prediction.metadata["scale"][0]), 1e-6)
                    covered = float(lower <= outcome <= upper)
                    history_scores.append(score)
                    history_cov.append(covered)
                    history_weights.append(weight)
                    ess = float((np.sum(history_weights) ** 2) / max(np.sum(np.square(history_weights)), 1e-6))
                    logger.log(
                        RolloutRecord(
                            dataset=f"D0_offpolicy:{overlap}",
                            method=method_name,
                            seed=seed,
                            round=round_idx,
                            context_id=int(deployed["context_id"]),
                            action=behavior_action,
                            outcome=outcome,
                            predicted_center=center,
                            set_lower_or_summary=lower,
                            set_upper_or_summary=upper,
                            set_volume=float(prediction.volume[0]),
                            score=score,
                            tau=float(method.tau),
                            covered_pre=float(lower <= center <= upper),
                            covered_post=covered,
                            fp_residual=float(abs(score - method.tau)),
                            risk_value=float(target_decision.risk_value[0]),
                            regret=float(deployed["oracle_regret"]),
                            is_on_policy=0 if method_name != "PCRC" else 1,
                            propensity=propensity,
                            importance_weight=weight,
                            ESS=ess,
                            geometry_type=method.geometry_type,
                            temperature=float(method.temperature),
                            surrogate_misspec_level=0.0,
                        )
                    )
                    weights = np.asarray(history_weights[-128:], dtype=float)
                    scores = np.asarray(history_scores[-128:], dtype=float)
                    cov = np.asarray(history_cov[-128:], dtype=float)
                    if method_name == "PCRC":
                        method.online_update(scores, cov)
                    else:
                        method.offpolicy_update(scores, cov, weights)


def _run_d2_component(config: ExperimentConfig, logger: RolloutLogger) -> None:
    archive = download_credit_dataset()
    frame = _pseudo_temporal_order(_prepare_credit_frame(load_credit_frame(archive)))
    feature_cols = [col for col in frame.columns if col not in {"default_next_month", "pseudo_time_score", "batch_index"}]
    train_end = int(len(frame) * 0.6)
    calib_end = int(len(frame) * 0.75)
    train = frame.iloc[:train_end].copy()
    calib = frame.iloc[train_end:calib_end].copy()
    deploy = frame.iloc[calib_end:].copy()
    kernel = CreditResponseKernel(
        lgd=float(config.params.get("lgd", 0.75)),
        interest_margin=float(config.params.get("interest_margin", 0.03)),
        approval_floor=float(config.params.get("approval_floor", 0.15)),
        composition_strength=float(config.params.get("composition_strength", 0.25)),
        near_threshold_band=float(config.params.get("near_threshold_band", 0.05)),
    )
    batch_size = int(config.params.get("batch_size", 256))
    beta = float(config.params.get("beta", 0.9))
    for seed in config.seeds:
        set_global_seed(seed)
        train_x = train[feature_cols].to_numpy(dtype=float)
        train_y = train["default_next_month"].to_numpy(dtype=float)
        calib_x = calib[feature_cols].to_numpy(dtype=float)
        calib_y = calib["default_next_month"].to_numpy(dtype=float)
        for overlap in config.params.get("overlaps", ["good", "medium", "bad"]):
            for method_name in config.methods:
                set_global_seed(seed)
                method = build_method(
                    method_name,
                    config.action_grid,
                    config.nominal_alpha,
                    str(config.params.get("model_kind", "credit_classifier")),
                    seed,
                    config.device,
                    beta=beta,
                    temperature=float(config.params.get("temperature", 0.1)),
                    method_overrides=_resolve_component_method_overrides(config, "D2", method_name),
                )
                method.fit_base_predictor(train_x, train_y)
                method.fit_uncertainty_module(calib_x, calib_y)
                current_pool = deploy.copy()
                weight_hist: list[float] = []
                cov_hist: list[float] = []
                score_hist: list[float] = []
                for round_idx in range(min(config.rounds, max(1, len(current_pool) // batch_size))):
                    batch = current_pool.iloc[:batch_size].copy()
                    x = batch[feature_cols].to_numpy(dtype=float)
                    prediction = method.predict_distribution_or_set(x)
                    center_p = np.clip(prediction.center, 1e-4, 1.0 - 1e-4)
                    exposure = batch["LIMIT_BAL"].to_numpy(dtype=float)
                    actual_individual_loss = kernel.individual_loss(batch)
                    center_loss, var_loss = _loss_moments(batch, center_p, kernel)
                    lower_loss, _ = _loss_moments(batch, np.clip(prediction.lower, 1e-4, 1.0 - 1e-4), kernel)
                    upper_loss, _ = _loss_moments(batch, np.clip(prediction.upper, 1e-4, 1.0 - 1e-4), kernel)
                    loss_scale = np.sqrt(np.clip(var_loss, 1e-6, None))
                    candidate_costs = {}
                    oracle_risks = {}
                    for threshold in config.action_grid:
                        approved_mask = center_p <= kernel.enforce_approval_floor(float(threshold), center_p)
                        if approved_mask.sum() == 0:
                            approved_mask = np.zeros_like(center_p, dtype=bool)
                            approved_mask[np.argsort(center_p)[: max(1, int(kernel.approval_floor * len(center_p)))]] = True
                        candidate_costs[float(threshold)] = _candidate_cost_draws(
                            center_loss=center_loss,
                            lower_loss=lower_loss,
                            upper_loss=upper_loss,
                            loss_scale=loss_scale,
                            mask=approved_mask,
                            tau=float(method.tau),
                            mc_points=int(config.params.get("mc_points", 32)),
                        )
                        oracle_risks[float(threshold)] = _tail_mean(actual_individual_loss[approved_mask], beta=beta)
                    target_decision = method.select_action(prediction, candidate_costs=candidate_costs)
                    target_threshold = float(target_decision.action[0])
                    behavior_probs = _soft_behavior_probs(config.action_grid, preferred=0.3, overlap=overlap)
                    behavior_threshold = _sample_behavior_action(
                        config.action_grid,
                        behavior_probs,
                        seed=seed,
                        component="D2",
                        overlap=overlap,
                        round_idx=round_idx,
                    )
                    propensity = float(behavior_probs[config.action_grid.index(behavior_threshold)])
                    target_propensity = float(_soft_behavior_probs(config.action_grid, preferred=target_threshold, overlap="good")[config.action_grid.index(behavior_threshold)])
                    weight = target_propensity / max(propensity, 1e-6)
                    if method_name == "PCRC":
                        weight = 1.0
                    threshold = kernel.enforce_approval_floor(behavior_threshold, center_p)
                    approved_mask = center_p <= threshold
                    if approved_mask.sum() == 0:
                        approved_mask = np.zeros_like(center_p, dtype=bool)
                        approved_mask[np.argsort(center_p)[: max(1, int(kernel.approval_floor * len(center_p)))]] = True
                    actual_losses = actual_individual_loss[approved_mask]
                    portfolio_loss = float(actual_losses.mean())
                    pred_center, pred_lower, pred_upper, pred_scale = _aggregate_interval(
                        center_loss=center_loss,
                        lower_loss=lower_loss,
                        upper_loss=upper_loss,
                        loss_scale=loss_scale,
                        mask=approved_mask,
                        tau=float(method.tau),
                    )
                    score = abs(portfolio_loss - pred_center) / max(pred_scale, 1e-6)
                    covered = float(pred_lower <= portfolio_loss <= pred_upper)
                    score_hist.append(score)
                    cov_hist.append(covered)
                    weight_hist.append(weight)
                    ess = float((np.sum(weight_hist) ** 2) / max(np.sum(np.square(weight_hist)), 1e-6))
                    logger.log(
                        RolloutRecord(
                            dataset=f"D2_offpolicy:{overlap}",
                            method=method_name,
                            seed=seed,
                            round=round_idx,
                            context_id=round_idx,
                            action=behavior_threshold,
                            outcome=portfolio_loss,
                            predicted_center=pred_center,
                            set_lower_or_summary=pred_lower,
                            set_upper_or_summary=pred_upper,
                            set_volume=float(pred_upper - pred_lower),
                            score=score,
                            tau=float(method.tau),
                            covered_pre=float(pred_lower <= pred_center <= pred_upper),
                            covered_post=covered,
                            fp_residual=float(abs(score - method.tau)),
                            risk_value=_tail_mean(actual_losses, beta=beta),
                            regret=float(oracle_risks[threshold] - min(oracle_risks.values())),
                            is_on_policy=0 if method_name != "PCRC" else 1,
                            propensity=propensity,
                            importance_weight=weight,
                            ESS=ess,
                            geometry_type=method.geometry_type,
                            temperature=float(method.temperature),
                            surrogate_misspec_level=0.0,
                        )
                    )
                    if method_name == "PCRC":
                        method.online_update(np.asarray(score_hist[-128:]), np.asarray(cov_hist[-128:]))
                    else:
                        method.offpolicy_update(np.asarray(score_hist[-128:]), np.asarray(cov_hist[-128:]), np.asarray(weight_hist[-128:]))
                    refresh_source = current_pool.iloc[batch_size:].copy()
                    if refresh_source.empty:
                        refresh_source = deploy.copy()
                    refresh_x = refresh_source[feature_cols].to_numpy(dtype=float)
                    refresh_pred = method.predict_distribution_or_set(refresh_x)
                    refresh_source["pred_default"] = np.clip(refresh_pred.center, 1e-4, 1.0 - 1e-4)
                    current_pool, _ = kernel.reweight_next_pool(
                        refresh_source,
                        threshold,
                        approval_rate=float(np.mean(approved_mask)),
                        random_state=seed * 1000 + round_idx,
                    )


def run_offpolicy_suite(config: ExperimentConfig) -> OffPolicyOutputs:
    logger = RolloutLogger()
    formal_run = is_formal_experiment(config.name)
    figure_dir = figure_dir_for_run(config.name)
    if "D0" in config.params.get("datasets", ["D0"]):
        _run_d0_component(config, logger)
    if "D2" in config.params.get("datasets", []):
        _run_d2_component(config, logger)
    frame = logger.to_frame()
    logger.save(config.name)
    summary = summarize_metrics(frame, ["dataset", "method"], metrics=list(DEFAULT_OFFPOLICY_METRICS))
    ranked = apply_coverage_priority(
        summary,
        nominal_alpha=config.nominal_alpha,
        mean_tolerance=0.02,
        ci_low_tolerance=0.05,
    )
    ranked["importance_weight_deviation"] = np.abs(ranked["importance_weight_mean"] - 1.0)
    gate_rank = {"pass": 0, "borderline": 1, "fail": 2}
    ranked["_gate_rank"] = ranked["coverage_gate"].map(gate_rank).fillna(9)
    ranked = ranked.sort_values(
        [
            "dataset",
            "_gate_rank",
            "mean_shortfall",
            "ci_shortfall",
            "risk_value_mean",
            "importance_weight_deviation",
            "ESS_mean",
        ],
        ascending=[True, True, True, True, True, True, False],
    ).reset_index(drop=True)
    ranked["coverage_priority_rank"] = ranked.groupby("dataset").cumcount() + 1
    summary_table = manuscript_table_from_ranked(ranked.drop(columns=["_gate_rank", "importance_weight_deviation"]), "table5_1")
    paired_tests = pairwise_method_tests(
        frame,
        metrics=["covered_post", "ESS", "importance_weight", "risk_value"],
        reference_method="PCRC",
        comparison_methods=[method for method in config.methods if method != "PCRC"],
        unit_cols=["dataset", "seed", "round"],
    )
    if formal_run:
        export_numbered_table(summary_table, "table5_1", "offpolicy_summary")
    else:
        export_table(summary_table, artifact_stem(config.name, "table5_1_offpolicy_summary"), table_dir=TABLES_APP_DIR)
    export_table(paired_tests, f"{config.name}_paired_tests", table_dir=TABLES_APP_DIR)
    export_table(summary, f"{config.name}_raw_summary", table_dir=TABLES_APP_DIR)
    plot_time_series(
        frame,
        "covered_post",
        artifact_stem(config.name, "fig5_1_offpolicy_coverage"),
        "Figure 5-1: Off-policy post-decision coverage",
        figure_dir=figure_dir,
    )
    plot_scatter(
        frame.assign(coverage_error=(1.0 - config.nominal_alpha) - frame["covered_post"]),
        "ESS",
        "coverage_error",
        artifact_stem(config.name, "fig5_2_ess_vs_error"),
        "Figure 5-2: ESS vs coverage error",
        hue="dataset",
        figure_dir=figure_dir,
    )
    plot_scatter(
        frame,
        "importance_weight",
        "ESS",
        artifact_stem(config.name, "fig5_3_weight_distribution"),
        "Figure 5-3: Off-policy weight diagnostics",
        hue="dataset",
        figure_dir=figure_dir,
    )
    write_results_tex("Off-Policy Results", summary_table, results_tex_path_for_run(config.name))
    return OffPolicyOutputs(rollout_frame=frame, summary_frame=summary)


def run_offpolicy_d0(config: ExperimentConfig) -> OffPolicyOutputs:
    return run_offpolicy_suite(config)
