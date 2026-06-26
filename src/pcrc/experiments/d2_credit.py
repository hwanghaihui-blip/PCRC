"""D2 UCI Credit paper-grade semi-synthetic experiment runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcrc.config import ExperimentConfig
from pcrc.constants import PROCESSED_DATA_DIR, TABLES_APP_DIR
from pcrc.data.credit import CreditResponseKernel, download_credit_dataset, load_credit_frame
from pcrc.io import artifact_stem, figure_dir_for_run, is_formal_experiment, results_tex_path_for_run
from pcrc.logging_utils import RolloutLogger, RolloutRecord
from pcrc.methods.conformal import build_method
from pcrc.reporting import (
    build_numbered_summary_table,
    export_numbered_table,
    export_table,
    pairwise_method_tests,
    plot_distribution,
    plot_scatter,
    plot_time_series,
    summarize_metrics,
    write_results_tex,
)
from pcrc.utils import save_frame, set_global_seed


def _prepare_credit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for col in prepared.columns:
        if prepared[col].dtype == "object":
            prepared[col] = prepared[col].astype("category").cat.codes
    prepared = prepared.reset_index(drop=True)
    return prepared


def _pseudo_temporal_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    pay_cols = [col for col in ordered.columns if col.startswith("PAY_")]
    bill_cols = [col for col in ordered.columns if col.startswith("BILL_AMT")]
    pay_amt_cols = [col for col in ordered.columns if col.startswith("PAY_AMT")]
    ordered["pseudo_time_score"] = (
        ordered[pay_cols].sum(axis=1)
        + 0.00001 * ordered[bill_cols].sum(axis=1)
        - 0.00001 * ordered[pay_amt_cols].sum(axis=1)
        + 0.001 * ordered["AGE"]
    )
    ordered = ordered.sort_values(["pseudo_time_score", "LIMIT_BAL", "AGE"]).reset_index(drop=True)
    ordered["batch_index"] = np.arange(len(ordered))
    return ordered


def _random_temporal_order(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    ordered = frame.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True).copy()
    ordered["pseudo_time_score"] = np.arange(len(ordered), dtype=float)
    ordered["batch_index"] = np.arange(len(ordered))
    return ordered


@dataclass
class D2ExperimentOutputs:
    rollout_frame: pd.DataFrame
    summary_frame: pd.DataFrame
    summary_table: pd.DataFrame
    slice_frame: pd.DataFrame
    frontier_frame: pd.DataFrame
    near_threshold_frame: pd.DataFrame
    composition_frame: pd.DataFrame
    frame_rows: int


def _tail_mean(values: np.ndarray, beta: float) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return float("nan")
    tail_count = max(1, int(np.ceil((1.0 - beta) * len(arr))))
    return float(arr[-tail_count:].mean())


def _loss_moments(frame: pd.DataFrame, probs: np.ndarray, kernel: CreditResponseKernel) -> tuple[np.ndarray, np.ndarray]:
    exposure = frame["LIMIT_BAL"].to_numpy(dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-5, 1.0 - 1e-5)
    pay_cols = [col for col in frame.columns if str(col).startswith("PAY_")]
    bill_cols = [col for col in frame.columns if str(col).startswith("BILL_AMT")]
    delinquency = np.clip(frame[pay_cols].to_numpy(dtype=float), 0.0, None).mean(axis=1) if pay_cols else np.zeros(len(frame), dtype=float)
    utilization = np.clip(frame[bill_cols].to_numpy(dtype=float), 0.0, None).mean(axis=1) / np.clip(exposure, 1.0, None) if bill_cols else np.zeros(len(frame), dtype=float)
    young_borrower = (frame["AGE"].to_numpy(dtype=float) < 30.0).astype(float)
    loss_multiplier = 1.0 + 0.08 * delinquency + 0.05 * utilization + 0.03 * young_borrower
    safe_gain = kernel.interest_margin * np.clip(1.0 - 0.10 * delinquency, 0.4, None)
    lift = kernel.lgd * loss_multiplier + safe_gain
    mean_loss = exposure * (kernel.lgd * loss_multiplier * probs - safe_gain * (1.0 - probs))
    var_loss = np.square(exposure * lift) * probs * (1.0 - probs)
    return mean_loss, var_loss


def _aggregate_interval(
    center_loss: np.ndarray,
    lower_loss: np.ndarray,
    upper_loss: np.ndarray,
    loss_scale: np.ndarray,
    mask: np.ndarray,
    tau: float,
) -> tuple[float, float, float, float]:
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 1 or selected.size == 0 or not selected.any():
        return float("nan"), float("nan"), float("nan"), float("nan")
    n = int(selected.sum())
    center = float(np.mean(np.asarray(center_loss, dtype=float)[selected]))
    scale = float(np.sqrt(np.square(loss_scale[selected]).sum()) / max(n, 1))
    prob_lower = float(np.mean(np.asarray(lower_loss, dtype=float)[selected]))
    prob_upper = float(np.mean(np.asarray(upper_loss, dtype=float)[selected]))
    lower = min(center - float(tau) * scale, prob_lower)
    upper = max(center + float(tau) * scale, prob_upper)
    return center, lower, upper, scale


def _candidate_cost_draws(
    center_loss: np.ndarray,
    lower_loss: np.ndarray,
    upper_loss: np.ndarray,
    loss_scale: np.ndarray,
    mask: np.ndarray,
    tau: float,
    mc_points: int = 32,
) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 1 or selected.size == 0 or not selected.any():
        return np.full(int(mc_points), np.nan, dtype=float)
    center, interval_low, interval_high, scale = _aggregate_interval(
        center_loss=center_loss,
        lower_loss=lower_loss,
        upper_loss=upper_loss,
        loss_scale=loss_scale,
        mask=selected,
        tau=tau,
    )
    support_low = float(interval_low)
    support_high = float(interval_high)
    if not np.isfinite(support_low) or not np.isfinite(support_high):
        return np.full(int(mc_points), np.nan, dtype=float)
    if support_high <= support_low:
        half_width = max(scale, 1e-6)
        support_low = center - half_width
        support_high = center + half_width
    return np.linspace(support_low, support_high, int(mc_points), dtype=float)


def _compute_frontier(details: pd.DataFrame, beta: float) -> pd.DataFrame:
    rows = []
    for method, group in details.groupby("method"):
        losses = group["portfolio_loss"].astype(float).to_numpy()
        rows.append(
            {
                "method": method,
                "approval_rate": float(group["approval_rate"].mean()),
                "mean_loss": float(losses.mean()),
                "cvar_loss": _tail_mean(losses, beta=beta),
                "near_threshold_coverage": float(group["near_threshold_coverage"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["approval_rate", "cvar_loss"])


def _slice_summary(details: pd.DataFrame) -> pd.DataFrame:
    details = details.copy()
    for column in ["set_size_or_volume", "fp_residual"]:
        if column not in details.columns:
            details[column] = np.nan
    summary = (
        details.groupby(["method", "slice_name"], as_index=False)
        .agg(
            approval_rate=("approval_rate", "mean"),
            C_post=("C_post", "mean"),
            C_pre=("C_pre", "mean"),
            cvar_loss=("cvar_loss", "mean"),
            set_size_or_volume=("set_size_or_volume", "mean"),
            portfolio_loss=("portfolio_loss", "mean"),
            regret=("regret", "mean"),
            fp_residual=("fp_residual", "mean"),
            runtime_or_iteration_count=("round", "size"),
        )
        .sort_values(["method", "slice_name"])
    )
    summary["gap"] = summary["C_pre"] - summary["C_post"]
    summary["risk(CVaR)"] = summary["cvar_loss"]
    return summary


def _build_slice_records(
    *,
    slice_masks: dict[str, np.ndarray],
    approved_mask: np.ndarray,
    anchor_mask: np.ndarray,
    candidate_masks: dict[float, np.ndarray],
    actual_individual_loss: np.ndarray,
    center_loss: np.ndarray,
    lower_loss: np.ndarray,
    upper_loss: np.ndarray,
    loss_scale: np.ndarray,
    tau: float,
    beta: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for slice_name, base_mask in slice_masks.items():
        base = np.asarray(base_mask, dtype=bool)
        population = int(base.sum())
        if population == 0:
            continue
        approved_slice = approved_mask & base
        anchor_slice = anchor_mask & base
        approval_rate = float(approved_slice.sum() / max(population, 1))
        portfolio_loss = cvar_loss = covered_post = set_size_or_volume = fp_residual = float("nan")
        if approved_slice.any():
            pred_center, pred_lower, pred_upper, pred_scale = _aggregate_interval(
                center_loss=center_loss,
                lower_loss=lower_loss,
                upper_loss=upper_loss,
                loss_scale=loss_scale,
                mask=approved_slice,
                tau=tau,
            )
            portfolio_loss = float(actual_individual_loss[approved_slice].mean())
            cvar_loss = _tail_mean(actual_individual_loss[approved_slice], beta=beta)
            covered_post = float(pred_lower <= portfolio_loss <= pred_upper)
            set_size_or_volume = float(pred_upper - pred_lower)
            slice_score = abs(portfolio_loss - pred_center) / max(float(pred_scale), 1e-6)
            fp_residual = float(abs(slice_score - tau))
        covered_pre = float("nan")
        if anchor_slice.any():
            _, anchor_lower, anchor_upper, _ = _aggregate_interval(
                center_loss=center_loss,
                lower_loss=lower_loss,
                upper_loss=upper_loss,
                loss_scale=loss_scale,
                mask=anchor_slice,
                tau=tau,
            )
            anchor_loss = float(actual_individual_loss[anchor_slice].mean())
            covered_pre = float(anchor_lower <= anchor_loss <= anchor_upper)
        candidate_risks = []
        for candidate_mask in candidate_masks.values():
            slice_candidate = np.asarray(candidate_mask, dtype=bool) & base
            if slice_candidate.any():
                candidate_risks.append(_tail_mean(actual_individual_loss[slice_candidate], beta=beta))
        regret = float("nan")
        if np.isfinite(cvar_loss) and candidate_risks:
            regret = float(cvar_loss - min(candidate_risks))
        rows.append(
            {
                "slice_name": slice_name,
                "approval_rate": approval_rate,
                "C_post": covered_post,
                "C_pre": covered_pre,
                "cvar_loss": cvar_loss,
                "set_size_or_volume": set_size_or_volume,
                "portfolio_loss": portfolio_loss,
                "regret": regret,
                "fp_residual": fp_residual,
            }
        )
    return rows


def run_d2_credit_experiment(config: ExperimentConfig) -> D2ExperimentOutputs:
    formal_run = is_formal_experiment(config.name)
    figure_dir = figure_dir_for_run(config.name)
    archive = download_credit_dataset()
    prepared_frame = _prepare_credit_frame(load_credit_frame(archive))
    ordering = str(config.params.get("temporal_ordering", "pseudo")).lower()
    if ordering == "random":
        frame = _random_temporal_order(prepared_frame, seed=int(config.params.get("temporal_ordering_seed", 314159)))
    else:
        frame = _pseudo_temporal_order(prepared_frame)
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
    pay_cols = [col for col in frame.columns if str(col).startswith("PAY_")]
    overdue_reference = np.clip(frame[pay_cols].to_numpy(dtype=float), 0.0, None).mean(axis=1) if pay_cols else np.zeros(len(frame), dtype=float)
    limit_median = float(frame["LIMIT_BAL"].median())
    age_median = float(frame["AGE"].median())
    overdue_median = float(np.median(overdue_reference)) if len(overdue_reference) else 0.0
    logger = RolloutLogger()
    details_rows: list[dict] = []
    slice_rows: list[dict] = []
    composition_rows: list[dict] = []
    method_overrides = config.params.get("method_overrides", {})
    for seed in config.seeds:
        set_global_seed(seed)
        train_x = train[feature_cols].to_numpy(dtype=float)
        train_y = train["default_next_month"].to_numpy(dtype=float)
        calib_x = calib[feature_cols].to_numpy(dtype=float)
        calib_y = calib["default_next_month"].to_numpy(dtype=float)
        for method_name in config.methods:
            set_global_seed(seed)
            method = build_method(
                method_name=method_name,
                action_grid=config.action_grid,
                nominal_alpha=config.nominal_alpha,
                model_kind=str(config.params.get("model_kind", "credit_classifier")),
                seed=seed,
                device=config.device,
                beta=beta,
                temperature=float(config.params.get("temperature", 0.1)),
                method_overrides=method_overrides.get(method_name, {}),
            )
            method.fit_base_predictor(train_x, train_y)
            method.fit_uncertainty_module(calib_x, calib_y)
            current_pool = deploy.copy()
            max_rounds = min(config.rounds, max(1, len(current_pool) // batch_size))
            for round_idx in range(max_rounds):
                batch = current_pool.iloc[:batch_size].copy()
                x = batch[feature_cols].to_numpy(dtype=float)
                prediction = method.predict_distribution_or_set(x)
                exposure = batch["LIMIT_BAL"].to_numpy(dtype=float)
                center_p = np.clip(prediction.center, 1e-4, 1.0 - 1e-4)
                lower_p = np.clip(prediction.lower, 1e-4, 1.0 - 1e-4)
                upper_p = np.clip(prediction.upper, 1e-4, 1.0 - 1e-4)
                batch["pred_default"] = center_p
                batch["risk_bin"] = kernel.assign_risk_bins(center_p)
                candidate_costs: dict[float, np.ndarray] = {}
                oracle_risks = {}
                approval_rates = {}
                candidate_masks: dict[float, np.ndarray] = {}
                actual_individual_loss = kernel.individual_loss(batch)
                center_loss, var_loss = _loss_moments(batch, center_p, kernel)
                lower_loss, _ = _loss_moments(batch, lower_p, kernel)
                upper_loss, _ = _loss_moments(batch, upper_p, kernel)
                loss_scale = np.sqrt(np.clip(var_loss, 1e-6, None))
                for threshold in config.action_grid:
                    clipped_threshold = kernel.enforce_approval_floor(float(threshold), center_p)
                    approved_mask = center_p <= clipped_threshold
                    if approved_mask.sum() == 0:
                        approved_mask = np.zeros_like(center_p, dtype=bool)
                        approved_mask[np.argsort(center_p)[: max(1, int(kernel.approval_floor * len(center_p)))]] = True
                    candidate_masks[float(threshold)] = approved_mask.copy()
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
                    approval_rates[float(threshold)] = float(np.mean(approved_mask))
                decision = method.select_action(prediction, candidate_costs=candidate_costs)
                selected_action = float(decision.action[0])
                threshold = kernel.enforce_approval_floor(selected_action, center_p)
                approved_mask = candidate_masks[selected_action].copy()
                approved = batch.loc[approved_mask].copy()
                actual_losses = actual_individual_loss[approved_mask]
                portfolio_loss = float(actual_losses.mean())
                realized_cvar = _tail_mean(actual_losses, beta=beta)
                pred_center, pred_lower, pred_upper, pred_scale = _aggregate_interval(
                    center_loss=center_loss,
                    lower_loss=lower_loss,
                    upper_loss=upper_loss,
                    loss_scale=loss_scale,
                    mask=approved_mask,
                    tau=float(method.tau),
                )
                score = abs(portfolio_loss - pred_center) / max(pred_scale, 1e-6)
                anchor_threshold = kernel.enforce_approval_floor(float(config.params.get("anchor_threshold", 0.3)), center_p)
                anchor_mask = center_p <= anchor_threshold
                if anchor_mask.sum() == 0:
                    anchor_mask = np.zeros_like(center_p, dtype=bool)
                    anchor_mask[np.argsort(center_p)[: max(1, int(kernel.approval_floor * len(center_p)))]] = True
                common_boundary_mask = kernel.near_threshold_mask(center_p, anchor_threshold)
                anchor_loss = float(actual_individual_loss[anchor_mask].mean())
                covered_pre = float(pred_lower <= anchor_loss <= pred_upper)
                covered_post = float(pred_lower <= portfolio_loss <= pred_upper)
                near_mask = kernel.near_threshold_mask(center_p, threshold)
                batch_overdue = np.clip(batch[pay_cols].to_numpy(dtype=float), 0.0, None).mean(axis=1) if pay_cols else np.zeros(len(batch), dtype=float)
                slice_masks = {
                    "overall": np.ones(len(batch), dtype=bool),
                    "common_boundary": common_boundary_mask,
                    "near_threshold": near_mask,
                    "high_limit": batch["LIMIT_BAL"].to_numpy(dtype=float) >= limit_median,
                    "low_limit": batch["LIMIT_BAL"].to_numpy(dtype=float) < limit_median,
                    "high_overdue": batch_overdue >= overdue_median,
                    "low_overdue": batch_overdue < overdue_median,
                    "young": batch["AGE"].to_numpy(dtype=float) < age_median,
                    "older": batch["AGE"].to_numpy(dtype=float) >= age_median,
                }
                round_slice_rows = _build_slice_records(
                    slice_masks=slice_masks,
                    approved_mask=approved_mask,
                    anchor_mask=anchor_mask,
                    candidate_masks=candidate_masks,
                    actual_individual_loss=actual_individual_loss,
                    center_loss=center_loss,
                    lower_loss=lower_loss,
                    upper_loss=upper_loss,
                    loss_scale=loss_scale,
                    tau=float(method.tau),
                    beta=beta,
                )
                for record in round_slice_rows:
                    slice_rows.append(
                        {
                            "dataset": "D2:Credit",
                            "method": method_name,
                            "seed": seed,
                            "round": round_idx,
                            "threshold": threshold,
                            **record,
                        }
                    )
                near_slice = next((record for record in round_slice_rows if record["slice_name"] == "near_threshold"), None)
                near_threshold_coverage = float(near_slice["C_post"]) if near_slice is not None else float("nan")
                logger.log(
                    RolloutRecord(
                        dataset="D2:Credit",
                        method=method_name,
                        seed=seed,
                        round=round_idx,
                        context_id=round_idx,
                        action=threshold,
                        outcome=portfolio_loss,
                        predicted_center=pred_center,
                        set_lower_or_summary=pred_lower,
                        set_upper_or_summary=pred_upper,
                        set_volume=float(pred_upper - pred_lower),
                        score=score,
                        tau=float(method.tau),
                        covered_pre=covered_pre,
                        covered_post=covered_post,
                        fp_residual=float(abs(score - method.tau)),
                        risk_value=realized_cvar,
                        regret=float(oracle_risks[selected_action] - min(oracle_risks.values())),
                        is_on_policy=1,
                        propensity=1.0 / len(config.action_grid),
                        importance_weight=1.0,
                        ESS=1.0,
                        geometry_type=method.geometry_type,
                        temperature=float(method.temperature),
                        surrogate_misspec_level=0.0,
                    )
                )
                details_rows.append(
                    {
                        "dataset": "D2:Credit",
                        "method": method_name,
                        "seed": seed,
                        "round": round_idx,
                        "threshold": threshold,
                        "approval_rate": float(approved_mask.mean()),
                        "portfolio_loss": portfolio_loss,
                        "risk_value": realized_cvar,
                        "regret": float(oracle_risks[selected_action] - min(oracle_risks.values())),
                        "covered_pre": covered_pre,
                        "covered_post": covered_post,
                        "near_threshold_coverage": near_threshold_coverage,
                    }
                )
                refresh_source = current_pool.iloc[batch_size:].copy()
                if refresh_source.empty:
                    refresh_source = deploy.copy()
                refresh_x = refresh_source[feature_cols].to_numpy(dtype=float)
                refresh_pred = method.predict_distribution_or_set(refresh_x)
                refresh_source["pred_default"] = np.clip(refresh_pred.center, 1e-4, 1.0 - 1e-4)
                current_pool, composition = kernel.reweight_next_pool(
                    refresh_source,
                    threshold,
                    approval_rate=float(approved_mask.mean()),
                    random_state=seed * 1000 + round_idx,
                )
                composition["method"] = method_name
                composition["seed"] = seed
                composition["round"] = round_idx
                composition_rows.extend(composition.to_dict(orient="records"))
                near_share = float(np.mean(near_mask)) if near_mask.any() else 0.0
                signal = 1.0 + 2.0 * near_share + abs(float(threshold) - float(config.params.get("anchor_threshold", 0.3)))
                method.online_update(np.asarray([score]), np.asarray([covered_post]), signals=np.asarray([signal], dtype=float))
    result = logger.to_frame()
    logger.save(config.name)
    details = pd.DataFrame(details_rows)
    slice_details = pd.DataFrame(slice_rows)
    composition_frame = pd.DataFrame(composition_rows)
    save_frame(details, PROCESSED_DATA_DIR / f"{config.name}_details.parquet")
    save_frame(slice_details, PROCESSED_DATA_DIR / f"{config.name}_slice_details.parquet")
    save_frame(composition_frame, PROCESSED_DATA_DIR / f"{config.name}_composition.parquet")
    summary = summarize_metrics(result, ["dataset", "method"])
    summary_table = build_numbered_summary_table(summary, "table4_1", nominal_alpha=config.nominal_alpha)
    frontier = _compute_frontier(details, beta=beta)
    slice_frame = _slice_summary(slice_details)
    near_threshold = slice_frame[slice_frame["slice_name"] == "near_threshold"].copy().reset_index(drop=True)
    paired_tests = pairwise_method_tests(
        result,
        metrics=["covered_post", "risk_value", "regret", "set_volume"],
        reference_method="PCRC",
        comparison_methods=[method for method in config.methods if method != "PCRC"],
        unit_cols=["dataset", "seed", "round"],
    )
    if formal_run:
        export_numbered_table(summary_table, "table4_1", "credit_overall_summary")
        export_numbered_table(slice_frame, "table4_2", "credit_slices")
    else:
        export_table(summary_table, artifact_stem(config.name, "table4_1_credit_overall_summary"), table_dir=TABLES_APP_DIR)
        export_table(slice_frame, artifact_stem(config.name, "table4_2_credit_slices"), table_dir=TABLES_APP_DIR)
    export_table(paired_tests, f"{config.name}_paired_tests", table_dir=TABLES_APP_DIR)
    export_table(summary, f"{config.name}_raw_summary", table_dir=TABLES_APP_DIR)
    export_table(near_threshold, f"{config.name}_near_threshold_summary", table_dir=TABLES_APP_DIR)
    plot_time_series(
        result,
        "action",
        artifact_stem(config.name, "fig4_1_threshold_trajectory"),
        "Figure 4-1: Threshold evolution",
        figure_dir=figure_dir,
    )
    plot_scatter(
        frontier,
        "approval_rate",
        "cvar_loss",
        artifact_stem(config.name, "fig4_2_approval_cvar_frontier"),
        "Figure 4-2: Approval-CVaR frontier",
        hue="method",
        figure_dir=figure_dir,
    )
    plot_scatter(
        near_threshold.rename(columns={"C_post": "near_cov"}),
        "near_cov",
        "cvar_loss",
        artifact_stem(config.name, "fig4_3_near_threshold_coverage"),
        "Figure 4-3: Near-threshold post-decision coverage",
        hue="method",
        figure_dir=figure_dir,
    )
    plot_distribution(
        slice_details[slice_details["slice_name"] == "overall"],
        "portfolio_loss",
        artifact_stem(config.name, "fig4_4_tail_loss_distribution"),
        "Figure 4-4: Portfolio tail-loss distribution",
        hue="method",
        figure_dir=figure_dir,
        kind="kde",
    )
    if not composition_frame.empty:
        plot_time_series(
            composition_frame.rename(columns={"share": "risk_value"}),
            "risk_value",
            artifact_stem(config.name, "fig4_5_pool_composition"),
            "Figure 4-5: Applicant-pool composition shift",
            hue="risk_bin",
            figure_dir=figure_dir,
        )
    write_results_tex("D2 Credit Results", summary_table, results_tex_path_for_run(config.name))
    return D2ExperimentOutputs(
        rollout_frame=result,
        summary_frame=summary,
        summary_table=summary_table,
        slice_frame=slice_frame,
        frontier_frame=frontier,
        near_threshold_frame=near_threshold,
        composition_frame=composition_frame,
        frame_rows=len(frame),
    )
