"""D1 M5 paper-grade semi-synthetic experiment runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcrc.config import ExperimentConfig
from pcrc.constants import PROCESSED_DATA_DIR, TABLES_APP_DIR
from pcrc.data.m5 import M5ResponseKernel, build_m5_subset_panel, build_subset_manifest, download_m5_dataset
from pcrc.io import artifact_stem, figure_dir_for_run, is_formal_experiment, results_tex_path_for_run
from pcrc.logging_utils import RolloutLogger, RolloutRecord
from pcrc.methods.conformal import build_method
from pcrc.reporting import (
    build_numbered_summary_table,
    export_numbered_table,
    export_table,
    pairwise_method_tests,
    plot_heatmap,
    plot_method_coverage_panels,
    plot_metric_panels,
    plot_scatter,
    summarize_metrics,
    write_results_tex,
)
from pcrc.utils import save_frame, set_global_seed


def _encode_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    encoded = frame.copy()
    for col in columns:
        if encoded[col].dtype == "object":
            encoded[col] = encoded[col].astype("category").cat.codes
    return encoded


@dataclass
class D1ExperimentOutputs:
    rollout_frame: pd.DataFrame
    summary_frame: pd.DataFrame
    summary_table: pd.DataFrame
    slice_frame: pd.DataFrame
    frontier_frame: pd.DataFrame
    trajectory_frame: pd.DataFrame
    panel_rows: int


def _m5_feature_columns() -> list[str]:
    return [
        "sales_lag_1",
        "sales_lag_7",
        "rolling_mean_7",
        "rolling_mean_28",
        "sell_price",
        "wday",
        "month",
        "event_flag",
        "event_type_primary",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "item_id",
        "store_id",
        "cat_id",
        "state_id",
    ]


def _compute_frontier(details: pd.DataFrame, beta: float) -> pd.DataFrame:
    rows = []
    for method, group in details.groupby("method"):
        losses = group["realized_loss"].astype(float).to_numpy()
        tail_count = max(1, int(np.ceil((1.0 - beta) * len(losses))))
        cvar_loss = float(np.sort(losses)[-tail_count:].mean())
        rows.append(
            {
                "method": method,
                "mean_profit": float((-losses).mean()),
                "cvar_loss": cvar_loss,
                "mean_shortfall": float(group["shortfall"].mean()),
                "mean_set_volume": float(group["set_volume"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("cvar_loss")


def _build_slice_frame(details: pd.DataFrame) -> pd.DataFrame:
    summary = (
        details.groupby(["method", "elasticity_group", "state_id", "cat_id", "promo_flag", "event_flag"], as_index=False, observed=True)
        .agg(
            C_post=("covered_post", "mean"),
            C_pre=("covered_pre", "mean"),
            regret=("regret", "mean"),
            set_volume=("set_volume", "mean"),
            mean_profit=("realized_profit", "mean"),
            cvar_proxy=("risk_value", "mean"),
            fp_residual=("fp_residual", "mean"),
            runtime_or_iteration_count=("round", "size"),
        )
        .sort_values(["method", "elasticity_group", "state_id", "cat_id", "promo_flag", "event_flag"])
    )
    summary["gap"] = summary["C_pre"] - summary["C_post"]
    summary["set_size_or_volume"] = summary["set_volume"]
    summary["risk(CVaR)"] = summary["cvar_proxy"]
    return summary


def _select_case_study_series(deploy: pd.DataFrame) -> dict[str, str]:
    if deploy.empty:
        return {"high": "", "low": ""}
    grouped = (
        deploy.groupby(["series_key", "elasticity_group"], observed=True)
        .agg(
            elasticity=("elasticity", "mean"),
            n_rows=("series_key", "size"),
            promo_levels=("promo_flag", "nunique"),
            event_hits=("event_flag", "sum"),
        )
        .reset_index()
        .sort_values(["n_rows", "series_key"], ascending=[False, True])
    )
    representatives = {"high": "", "low": ""}
    for level, ascending in [("high", True), ("low", False)]:
        subset = grouped[grouped["elasticity_group"] == level].copy()
        if subset.empty:
            continue
        choice = subset.sort_values(
            ["n_rows", "event_hits", "promo_levels", "elasticity", "series_key"],
            ascending=[False, False, False, ascending, True],
        ).iloc[0]
        representatives[level] = str(choice["series_key"])
    return representatives


def _build_case_deploy_schedule(
    deploy: pd.DataFrame,
    rounds: int,
    trajectory_points_per_level: int = 12,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if deploy.empty or rounds <= 0:
        return deploy.head(0).copy(), {"high": "", "low": ""}
    ordered = deploy.sort_values(["date", "series_key"]).reset_index().rename(columns={"index": "source_index"})
    representatives = _select_case_study_series(ordered)
    selected_parts: list[pd.DataFrame] = []
    reserved_indices: set[int] = set()
    for level, series_key in representatives.items():
        if not series_key:
            continue
        series_rows = ordered[ordered["series_key"] == series_key].sort_values(["date", "source_index"]).head(trajectory_points_per_level).copy()
        if series_rows.empty:
            continue
        series_rows["selection_role"] = f"trajectory_{level}"
        selected_parts.append(series_rows)
        reserved_indices.update(int(value) for value in series_rows["source_index"].tolist())
    remaining_slots = max(0, rounds - sum(len(part) for part in selected_parts))
    remainder = ordered[~ordered["source_index"].isin(reserved_indices)].copy()
    if remaining_slots > 0 and not remainder.empty:
        groups = []
        for key, group in remainder.groupby(["promo_flag", "event_flag", "elasticity_group"], dropna=False, sort=False, observed=True):
            groups.append((key, group.sort_values(["date", "series_key", "source_index"]).copy()))
        groups.sort(key=lambda item: (len(item[1]), -int(item[0][1]), int(item[0][0]), str(item[0][2])))
        cursors = {key: 0 for key, _ in groups}
        diversity_rows: list[pd.DataFrame] = []
        for key, group in groups:
            if len(diversity_rows) >= remaining_slots:
                break
            diversity_rows.append(group.iloc[[0]].assign(selection_role="diversity"))
            cursors[key] = 1
        while len(diversity_rows) < remaining_slots:
            added = False
            for key, group in groups:
                position = cursors[key]
                if position >= len(group):
                    continue
                diversity_rows.append(group.iloc[[position]].assign(selection_role="diversity"))
                cursors[key] = position + 1
                added = True
                if len(diversity_rows) >= remaining_slots:
                    break
            if not added:
                break
        if diversity_rows:
            selected_parts.append(pd.concat(diversity_rows, ignore_index=True))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else ordered.head(rounds).copy()
    selected = selected.drop_duplicates(subset="source_index").sort_values(["date", "source_index"]).head(rounds).reset_index(drop=True)
    selected["trajectory_level"] = ""
    for level, series_key in representatives.items():
        if series_key:
            selected.loc[selected["series_key"] == series_key, "trajectory_level"] = level
    return selected, representatives


def run_d1_m5_experiment(config: ExperimentConfig, hf_token: str | None = None) -> D1ExperimentOutputs:
    formal_run = is_formal_experiment(config.name)
    figure_dir = figure_dir_for_run(config.name)
    paths = download_m5_dataset(token=hf_token)
    panel = build_m5_subset_panel(
        paths["calendar.csv"].parent,
        states=config.params.get("states"),
        categories=config.params.get("categories"),
        stores=config.params.get("stores"),
        max_rows=config.params.get("max_series", 400),
    )
    kernel = M5ResponseKernel(
        stockout_penalty=float(config.params.get("stockout_penalty", 0.8)),
        holding_cost=float(config.params.get("holding_cost", 0.05)),
        variance_scale=float(config.params.get("variance_scale", 0.35)),
    )
    manifest = build_subset_manifest(panel)
    elasticity = kernel.fit_elasticity(panel)
    panel = panel.merge(elasticity, on=["item_id", "store_id", "cat_id", "state_id"], how="left")
    panel = kernel.annotate_anchor_loss(panel)
    panel["elasticity"] = panel["elasticity"].fillna(kernel.beta_price)
    panel["elasticity_group"] = panel["elasticity_group"].fillna("mid")
    panel["elasticity"] = panel["elasticity"] * float(config.params.get("elasticity_multiplier", 1.0))
    series_keys = panel["item_id"].astype(str) + "::" + panel["store_id"].astype(str)
    panel["series_key"] = series_keys
    feature_cols = _m5_feature_columns()
    panel = _encode_features(panel, feature_cols)
    panel = panel.sort_values("date").reset_index(drop=True)
    train_end = int(len(panel) * 0.6)
    calib_end = int(len(panel) * 0.75)
    train = panel.iloc[:train_end].copy()
    calib = panel.iloc[train_end:calib_end].copy()
    deploy = panel.iloc[calib_end:].copy()
    deploy_schedule, representatives = _build_case_deploy_schedule(
        deploy,
        rounds=config.rounds,
        trajectory_points_per_level=int(config.params.get("trajectory_points_per_level", 12)),
    )
    save_frame(manifest, PROCESSED_DATA_DIR / "m5_subset_manifest.csv")
    save_frame(
        deploy[
            [
                "date",
                "item_id",
                "store_id",
                "cat_id",
                "state_id",
                "elasticity",
                "elasticity_group",
                "promo_flag",
                "event_flag",
                "series_key",
            ]
        ],
        PROCESSED_DATA_DIR / "m5_deploy_metadata.csv",
    )
    save_frame(
        deploy_schedule[
            [
                "source_index",
                "date",
                "item_id",
                "store_id",
                "cat_id",
                "state_id",
                "elasticity",
                "elasticity_group",
                "promo_flag",
                "event_flag",
                "series_key",
                "selection_role",
                "trajectory_level",
            ]
        ],
        PROCESSED_DATA_DIR / "m5_case_schedule.csv",
    )
    logger = RolloutLogger()
    details_rows: list[dict] = []
    beta = float(config.params.get("beta", 0.9))
    method_overrides = config.params.get("method_overrides", {})
    model_kind = str(config.params.get("model_kind", "lgbm_regression"))
    monte_carlo_draws = int(config.params.get("monte_carlo_draws", 32))
    for seed in config.seeds:
        set_global_seed(seed)
        train_x = train[feature_cols].to_numpy(dtype=float)
        train_y = train["anchor_loss"].to_numpy(dtype=float)
        calib_x = calib[feature_cols].to_numpy(dtype=float)
        calib_y = calib["anchor_loss"].to_numpy(dtype=float)
        for method_name in config.methods:
            set_global_seed(seed)
            method = build_method(
                method_name=method_name,
                action_grid=config.action_grid,
                nominal_alpha=config.nominal_alpha,
                model_kind=model_kind,
                seed=seed,
                device=config.device,
                beta=beta,
                temperature=float(config.params.get("temperature", 0.1)),
                method_overrides=method_overrides.get(method_name, {}),
            )
            method.fit_base_predictor(train_x, train_y)
            method.fit_uncertainty_module(calib_x, calib_y)
            for round_idx, (_, row) in enumerate(deploy_schedule.iterrows()):
                row_source_index = int(row["source_index"])
                x = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
                prediction = method.predict_distribution_or_set(x)
                candidate_costs: dict[float, np.ndarray] = {}
                oracle_risks = {}
                realized_draws: dict[float, dict[str, np.ndarray]] = {}
                for action in config.action_grid:
                    candidate_rng = kernel.deterministic_rng(seed, "candidate", row_source_index, round_idx, float(action))
                    sims = kernel.simulate_demand_batch(row, action, float(row["elasticity"]), candidate_rng, n_draws=monte_carlo_draws)
                    costs = np.asarray(sims["loss"], dtype=float)
                    realized_draws[action] = sims
                    candidate_costs[action] = costs
                    tail_count = max(1, int(np.ceil((1.0 - beta) * len(costs))))
                    oracle_risks[action] = float(np.sort(costs)[-tail_count:].mean())
                decision = method.select_action(prediction, candidate_costs=candidate_costs)
                action = float(decision.action[0])
                deployed_batch = kernel.simulate_demand_batch(
                    row,
                    action,
                    float(row["elasticity"]),
                    kernel.deterministic_rng(seed, "deploy", row_source_index, round_idx, action),
                    n_draws=1,
                )
                anchored_batch = kernel.simulate_demand_batch(
                    row,
                    1.0,
                    float(row["elasticity"]),
                    kernel.deterministic_rng(seed, "anchor", row_source_index, round_idx, 1.0),
                    n_draws=1,
                )
                center = float(prediction.center[0])
                lower = float(prediction.lower[0])
                upper = float(prediction.upper[0])
                outcome = float(deployed_batch["loss"][0])
                score = abs(outcome - center) / max(float(prediction.metadata["scale"][0]), 1e-6)
                covered_pre = float(lower <= float(anchored_batch["loss"][0]) <= upper)
                covered_post = float(lower <= outcome <= upper)
                realized_tail = np.asarray(realized_draws[action]["loss"], dtype=float)
                tail_count = max(1, int(np.ceil((1.0 - beta) * len(realized_tail))))
                realized_cvar = float(np.sort(realized_tail)[-tail_count:].mean())
                logger.log(
                    RolloutRecord(
                        dataset="D1:M5",
                        method=method_name,
                        seed=seed,
                        round=round_idx,
                        context_id=row_source_index,
                        action=action,
                        outcome=outcome,
                        predicted_center=center,
                        set_lower_or_summary=lower,
                        set_upper_or_summary=upper,
                        set_volume=float(prediction.volume[0]),
                        score=score,
                        tau=float(method.tau),
                        covered_pre=covered_pre,
                        covered_post=covered_post,
                        fp_residual=float(abs(score - method.tau)),
                        risk_value=realized_cvar,
                        regret=float(oracle_risks[action] - min(oracle_risks.values())),
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
                        "dataset": "D1:M5",
                        "method": method_name,
                        "seed": seed,
                        "round": round_idx,
                        "context_id": row_source_index,
                        "state_id": row["state_id"],
                        "cat_id": row["cat_id"],
                        "promo_flag": int(row["promo_flag"]),
                        "elasticity": float(row["elasticity"]),
                        "elasticity_group": row["elasticity_group"],
                        "series_key": row["series_key"],
                        "action": action,
                        "covered_pre": covered_pre,
                        "covered_post": covered_post,
                        "set_volume": float(prediction.volume[0]),
                        "risk_value": realized_cvar,
                        "regret": float(oracle_risks[action] - min(oracle_risks.values())),
                        "realized_loss": outcome,
                        "realized_profit": float(deployed_batch["profit"][0]),
                        "shortfall": float(deployed_batch["shortfall"][0]),
                        "price": float(deployed_batch["price"][0]),
                        "event_flag": int(row.get("event_flag", 0)),
                        "demand": float(deployed_batch["demand"][0]),
                        "fp_residual": float(abs(score - method.tau)),
                        "trajectory_level": row.get("trajectory_level", ""),
                    }
                )
                signal = 1.0 + abs(float(row["elasticity"])) * (1.0 + int(row.get("promo_flag", 0)) + int(row.get("event_flag", 0)))
                method.online_update(np.asarray([score]), np.asarray([covered_post]), signals=np.asarray([signal], dtype=float))
    frame = logger.to_frame()
    logger.save(config.name)
    details = pd.DataFrame(details_rows)
    save_frame(details, PROCESSED_DATA_DIR / f"{config.name}_details.parquet")
    summary = summarize_metrics(frame, ["dataset", "method"])
    summary_table = build_numbered_summary_table(summary, "table3_1", nominal_alpha=config.nominal_alpha)
    slice_frame = _build_slice_frame(details)
    frontier = _compute_frontier(details, beta=beta)
    paired_tests = pairwise_method_tests(
        frame,
        metrics=["covered_post", "risk_value", "regret", "set_volume"],
        reference_method="PCRC",
        comparison_methods=[method for method in config.methods if method != "PCRC"],
        unit_cols=["dataset", "seed", "round"],
    )
    trajectory_rows = []
    for level, series_key in representatives.items():
        if not series_key:
            continue
        subset = details[details["series_key"] == series_key].copy()
        if subset.empty:
            continue
        subset["trajectory_level"] = level
        subset = subset.sort_values(["seed", "method", "round"]).copy()
        subset["trajectory_step"] = subset.groupby(["seed", "method", "trajectory_level"]).cumcount() + 1
        trajectory_rows.append(subset)
    trajectory_frame = pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame(columns=details.columns.tolist() + ["trajectory_level"])
    save_frame(trajectory_frame, PROCESSED_DATA_DIR / f"{config.name}_trajectory.parquet")
    if formal_run:
        export_numbered_table(summary_table, "table3_1", "m5_overall_summary")
        export_numbered_table(slice_frame, "table3_2", "m5_slices")
    else:
        export_table(summary_table, artifact_stem(config.name, "table3_1_m5_overall_summary"), table_dir=TABLES_APP_DIR)
        export_table(slice_frame, artifact_stem(config.name, "table3_2_m5_slices"), table_dir=TABLES_APP_DIR)
    export_table(paired_tests, f"{config.name}_paired_tests", table_dir=TABLES_APP_DIR)
    export_table(summary, f"{config.name}_raw_summary", table_dir=TABLES_APP_DIR)
    plot_heatmap(
        manifest,
        "state_id",
        "cat_id",
        "n_items",
        artifact_stem(config.name, "fig3_1_m5_subset_layout"),
        "Figure 3-1: M5 stratified subset composition",
        figure_dir=figure_dir,
    )
    plot_method_coverage_panels(
        frame.melt(id_vars=["dataset", "method", "seed", "round"], value_vars=["covered_pre", "covered_post"], var_name="coverage_type", value_name="coverage_value"),
        artifact_stem(config.name, "fig3_2_m5_coverage_compare"),
        "Figure 3-2: M5 pre/post coverage comparison",
        hline=1.0 - config.nominal_alpha,
        figure_dir=figure_dir,
    )
    plot_scatter(
        frontier,
        "mean_profit",
        "cvar_loss",
        artifact_stem(config.name, "fig3_3_m5_profit_cvar_frontier"),
        "Figure 3-3: M5 profit-CVaR frontier",
        hue="method",
        figure_dir=figure_dir,
    )
    elasticity_summary = details.groupby(["method", "elasticity_group"], as_index=False)[["covered_post", "regret"]].mean()
    plot_metric_panels(
        elasticity_summary,
        "elasticity_group",
        ["covered_post", "regret"],
        artifact_stem(config.name, "fig3_4_m5_elasticity_groups"),
        "Figure 3-4: M5 elasticity-group coverage/regret comparison",
        hue="method",
        figure_dir=figure_dir,
        kind="bar",
        hlines={"covered_post": 1.0 - config.nominal_alpha},
        ylabels={"covered_post": "Post-decision coverage", "regret": "Regret"},
        metric_titles={"covered_post": "Coverage by Elasticity Group", "regret": "Regret by Elasticity Group"},
        x_label="Elasticity group",
    )
    if not trajectory_frame.empty:
        high = trajectory_frame[trajectory_frame["trajectory_level"] == "high"]
        low = trajectory_frame[trajectory_frame["trajectory_level"] == "low"]
        if not high.empty:
            plot_metric_panels(
                high,
                "trajectory_step",
                ["price", "demand", "set_volume", "covered_post", "risk_value", "fp_residual"],
                artifact_stem(config.name, "fig3_5_m5_high_elasticity_trajectory"),
                "Figure 3-5: High-elasticity representative trajectory",
                hue="method",
                figure_dir=figure_dir,
                kind="line",
                hlines={"covered_post": 1.0 - config.nominal_alpha, "fp_residual": 0.0},
                ylabels={
                    "price": "Price",
                    "demand": "Demand",
                    "set_volume": "Set width",
                    "covered_post": "Coverage",
                    "risk_value": "CVaR proxy",
                    "fp_residual": "Fixed-point residual",
                },
                metric_titles={
                    "price": "Price Action",
                    "demand": "Realized Demand",
                    "set_volume": "Uncertainty Width",
                    "covered_post": "Post-decision Coverage",
                    "risk_value": "Tail Risk",
                    "fp_residual": "Fixed-point Residual",
                },
                x_label="Series step",
            )
        if not low.empty:
            plot_metric_panels(
                low,
                "trajectory_step",
                ["price", "demand", "set_volume", "covered_post", "risk_value", "fp_residual"],
                artifact_stem(config.name, "fig3_6_m5_low_elasticity_trajectory"),
                "Figure 3-6: Low-elasticity representative trajectory",
                hue="method",
                figure_dir=figure_dir,
                kind="line",
                hlines={"covered_post": 1.0 - config.nominal_alpha, "fp_residual": 0.0},
                ylabels={
                    "price": "Price",
                    "demand": "Demand",
                    "set_volume": "Set width",
                    "covered_post": "Coverage",
                    "risk_value": "CVaR proxy",
                    "fp_residual": "Fixed-point residual",
                },
                metric_titles={
                    "price": "Price Action",
                    "demand": "Realized Demand",
                    "set_volume": "Uncertainty Width",
                    "covered_post": "Post-decision Coverage",
                    "risk_value": "Tail Risk",
                    "fp_residual": "Fixed-point Residual",
                },
                x_label="Series step",
            )
    write_results_tex("D1 M5 Results", summary_table, results_tex_path_for_run(config.name))
    return D1ExperimentOutputs(
        rollout_frame=frame,
        summary_frame=summary,
        summary_table=summary_table,
        slice_frame=slice_frame,
        frontier_frame=frontier,
        trajectory_frame=trajectory_frame,
        panel_rows=len(panel),
    )
