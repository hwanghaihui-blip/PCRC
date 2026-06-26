"""Reporting, statistical tables, figure helpers, and manuscript exports."""

from __future__ import annotations

from pathlib import Path
import re
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from pcrc.contract import method_label_map, table_columns
from pcrc.constants import FIGURES_APP_DIR, FIGURES_MAIN_DIR, TABLES_APP_DIR, TABLES_MAIN_DIR
from pcrc.utils import bootstrap_mean_ci, ensure_parent, latex_escape


DEFAULT_SUMMARY_METRICS = (
    "covered_pre",
    "covered_post",
    "set_volume",
    "risk_value",
    "regret",
    "fp_residual",
)


SUMMARY_COLUMN_ALIASES = {
    "covered_pre_mean": "C_pre",
    "covered_pre_std": "C_pre_std",
    "covered_pre_ci_low": "C_pre_ci_low",
    "covered_pre_ci_high": "C_pre_ci_high",
    "covered_post_mean": "C_post",
    "covered_post_std": "C_post_std",
    "covered_post_ci_low": "C_post_ci_low",
    "covered_post_ci_high": "C_post_ci_high",
    "gap_mean": "gap",
    "set_volume_mean": "set_size_or_volume",
    "set_volume_std": "set_size_or_volume_std",
    "set_volume_ci_low": "set_size_or_volume_ci_low",
    "set_volume_ci_high": "set_size_or_volume_ci_high",
    "risk_value_mean": "risk(CVaR)",
    "risk_value_std": "risk(CVaR)_std",
    "risk_value_ci_low": "risk(CVaR)_ci_low",
    "risk_value_ci_high": "risk(CVaR)_ci_high",
    "regret_mean": "regret",
    "regret_std": "regret_std",
    "regret_ci_low": "regret_ci_low",
    "regret_ci_high": "regret_ci_high",
    "fp_residual_mean": "fp_residual",
    "fp_residual_std": "fp_residual_std",
    "fp_residual_ci_low": "fp_residual_ci_low",
    "fp_residual_ci_high": "fp_residual_ci_high",
    "ESS_mean": "ESS",
    "ESS_ci_low": "ESS_ci_low",
    "ESS_ci_high": "ESS_ci_high",
}


CORE_TABLE_COLUMNS = (
    "method",
    "C_pre",
    "C_post",
    "gap",
    "set_size_or_volume",
    "risk(CVaR)",
    "regret",
    "fp_residual",
    "runtime_or_iteration_count",
)


def _display_method_labels(frame: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    if method_col not in frame.columns:
        return frame
    mapping = method_label_map()
    displayed = frame.copy()
    displayed[method_col] = displayed[method_col].map(lambda value: mapping.get(value, value))
    return displayed


def summarize_metrics(frame: pd.DataFrame, group_cols: list[str], metrics: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    selected_metrics = [metric for metric in (metrics or DEFAULT_SUMMARY_METRICS) if metric in frame.columns]
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in selected_metrics:
            values = group[metric].astype(float)
            mean, lower, upper = bootstrap_mean_ci(values.tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_ci_low"] = lower
            row[f"{metric}_ci_high"] = upper
        if "covered_pre_mean" in row and "covered_post_mean" in row:
            row["gap_mean"] = row["covered_pre_mean"] - row["covered_post_mean"]
        row["runtime_or_iteration_count"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def apply_coverage_priority(
    frame: pd.DataFrame,
    nominal_alpha: float,
    group_cols: list[str] | str = "dataset",
    mean_tolerance: float = 0.01,
    ci_low_tolerance: float = 0.03,
) -> pd.DataFrame:
    if frame.empty:
        enriched = frame.copy()
        enriched["coverage_gate"] = []
        enriched["coverage_priority_rank"] = []
        return enriched
    if "covered_post_mean" not in frame.columns or "covered_post_ci_low" not in frame.columns:
        raise ValueError("Coverage priority requires covered_post_mean and covered_post_ci_low.")
    keys = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    target = 1.0 - nominal_alpha
    eps = 1e-12
    enriched = frame.copy()
    enriched["coverage_target"] = target
    enriched["mean_shortfall"] = (target - enriched["covered_post_mean"]).clip(lower=0.0)
    enriched["ci_shortfall"] = (target - enriched["covered_post_ci_low"]).clip(lower=0.0)
    enriched["coverage_gate"] = "fail"
    pass_mask = (enriched["mean_shortfall"] <= mean_tolerance + eps) & (enriched["ci_shortfall"] <= ci_low_tolerance + eps)
    borderline_mask = (enriched["mean_shortfall"] <= mean_tolerance + 0.01 + eps) & (enriched["ci_shortfall"] <= ci_low_tolerance + 0.02 + eps)
    enriched.loc[borderline_mask, "coverage_gate"] = "borderline"
    enriched.loc[pass_mask, "coverage_gate"] = "pass"
    gate_rank = {"pass": 0, "borderline": 1, "fail": 2}
    enriched["_gate_rank"] = enriched["coverage_gate"].map(gate_rank).fillna(9)
    for metric in ["risk_value_mean", "regret_mean", "set_volume_mean"]:
        if metric not in enriched.columns:
            enriched[metric] = float("inf")
    enriched = enriched.sort_values(keys + ["_gate_rank", "mean_shortfall", "ci_shortfall", "risk_value_mean", "regret_mean", "set_volume_mean"]).reset_index(drop=True)
    enriched["coverage_priority_rank"] = enriched.groupby(keys).cumcount() + 1
    return enriched.drop(columns=["_gate_rank"])


def _fill_contract_columns(frame: pd.DataFrame, number: str) -> pd.DataFrame:
    normalized = frame.copy()
    if number == "table2_2":
        normalized["method"] = normalized.get("method", "PCRC")
        normalized["fp_residual"] = normalized.get("fp_residual", normalized.get("fp_residual_mean", "N/A"))
        normalized["runtime_or_iteration_count"] = normalized.get(
            "runtime_or_iteration_count",
            normalized.get("round_count", "N/A"),
        )
    elif number == "table2_3":
        normalized["method"] = normalized.get("method", "pooled")
        normalized["runtime_or_iteration_count"] = normalized.get(
            "runtime_or_iteration_count",
            normalized.get("n_samples", "N/A"),
        )
    elif number == "table3_2":
        if "gap" not in normalized.columns and {"C_pre", "C_post"}.issubset(normalized.columns):
            normalized["gap"] = normalized["C_pre"] - normalized["C_post"]
        if "set_size_or_volume" not in normalized.columns and "set_volume" in normalized.columns:
            normalized["set_size_or_volume"] = normalized["set_volume"]
        if "risk(CVaR)" not in normalized.columns and "cvar_proxy" in normalized.columns:
            normalized["risk(CVaR)"] = normalized["cvar_proxy"]
    elif number == "table4_2":
        if "gap" not in normalized.columns and {"C_pre", "C_post"}.issubset(normalized.columns):
            normalized["gap"] = normalized["C_pre"] - normalized["C_post"]
        if "risk(CVaR)" not in normalized.columns and "cvar_loss" in normalized.columns:
            normalized["risk(CVaR)"] = normalized["cvar_loss"]
    for column in CORE_TABLE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "N/A"
    if number in {"table2_2", "table2_3", "table3_2", "table4_2"}:
        for column in CORE_TABLE_COLUMNS:
            normalized[column] = normalized[column].where(pd.notna(normalized[column]), "N/A")
    return normalized


def coerce_table_schema(frame: pd.DataFrame, number: str) -> pd.DataFrame:
    expected = table_columns(number)
    if expected is None:
        return frame
    frame = _fill_contract_columns(frame, number)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Table {number} is missing required columns: {missing}")
    return frame.loc[:, list(expected)].copy()


def build_numbered_summary_table(
    summary_frame: pd.DataFrame,
    table_number: str,
    nominal_alpha: float,
    mean_tolerance: float = 0.01,
    ci_low_tolerance: float = 0.03,
    group_cols: list[str] | str = "dataset",
) -> pd.DataFrame:
    ranked = apply_coverage_priority(
        frame=summary_frame,
        nominal_alpha=nominal_alpha,
        group_cols=group_cols,
        mean_tolerance=mean_tolerance,
        ci_low_tolerance=ci_low_tolerance,
    )
    return manuscript_table_from_ranked(ranked, table_number)


def manuscript_table_from_ranked(frame: pd.DataFrame, table_number: str) -> pd.DataFrame:
    manuscript = frame.copy()
    if "method" not in manuscript.columns:
        manuscript["method"] = "PCRC"
    for source, target in SUMMARY_COLUMN_ALIASES.items():
        if source in manuscript.columns:
            manuscript[target] = manuscript[source]
    if "covered_post_mean" in manuscript.columns:
        manuscript["coverage"] = manuscript["covered_post_mean"]
        manuscript["coverage_ci_low"] = manuscript.get("covered_post_ci_low", np.nan)
        manuscript["coverage_ci_high"] = manuscript.get("covered_post_ci_high", np.nan)
    if "importance_weight_std" in manuscript.columns:
        manuscript["variance"] = np.square(manuscript["importance_weight_std"])
    elif "importance_weight_mean" in manuscript.columns and "variance" not in manuscript.columns:
        manuscript["variance"] = np.nan
    return coerce_table_schema(manuscript, table_number)


def export_table(frame: pd.DataFrame, stem: str, table_dir: Path | None = None, display_method_labels: bool = True) -> tuple[Path, Path]:
    destination = table_dir or TABLES_MAIN_DIR
    display_frame = _display_method_labels(frame) if display_method_labels else frame.copy()
    csv_path = ensure_parent(destination / f"{stem}.csv")
    tex_path = ensure_parent(destination / f"{stem}.tex")
    display_frame.to_csv(csv_path, index=False)
    tex_lines = ["\\begin{tabular}{" + "l" * len(display_frame.columns) + "}", "\\toprule"]
    tex_lines.append(" & ".join(latex_escape(col) for col in display_frame.columns) + " \\\\")
    tex_lines.append("\\midrule")
    for _, row in display_frame.iterrows():
        tex_lines.append(" & ".join(latex_escape(str(value)) for value in row.tolist()) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    return csv_path, tex_path


def export_numbered_table(
    frame: pd.DataFrame,
    number: str,
    stem: str,
    appendix: bool = False,
    display_method_labels: bool = True,
) -> tuple[Path, Path]:
    directory = TABLES_APP_DIR if appendix else TABLES_MAIN_DIR
    coerced = coerce_table_schema(frame, number)
    return export_table(coerced, f"{number}_{stem}", table_dir=directory, display_method_labels=display_method_labels)


def plot_time_series(
    frame: pd.DataFrame,
    y: str,
    stem: str,
    title: str,
    hue: str = "method",
    figure_dir: Path | None = None,
    hline: float | None = None,
) -> tuple[Path, Path]:
    plot_frame = _display_method_labels(frame, method_col=hue) if hue == "method" else frame
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=plot_frame, x="round", y=y, hue=hue, ax=ax, estimator="mean", errorbar=("ci", 95))
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Deployment round")
    ax.set_ylabel(y)
    target_dir = figure_dir or FIGURES_MAIN_DIR
    pdf_path = ensure_parent(target_dir / f"{stem}.pdf")
    png_path = ensure_parent(target_dir / f"{stem}.png")
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    return pdf_path, png_path


def plot_method_coverage_panels(
    frame: pd.DataFrame,
    stem: str,
    title: str,
    figure_dir: Path | None = None,
    hline: float | None = None,
    style: str | None = None,
) -> tuple[Path, Path]:
    plot_frame = _display_method_labels(frame)
    methods = [method for method in plot_frame["method"].dropna().unique().tolist()]
    if not methods:
        raise ValueError("plot_method_coverage_panels requires at least one method.")
    cols = 2 if len(methods) <= 4 else 3
    rows = int(math.ceil(len(methods) / float(cols)))
    sns.set_theme(style="white")
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.8 * rows), squeeze=False, sharex=True, sharey=True)
    handles = labels = None
    for ax, method in zip(axes.flat, methods):
        subset = plot_frame[plot_frame["method"] == method]
        sns.lineplot(
            data=subset,
            x="round",
            y="coverage_value",
            hue="coverage_type",
            style=style,
            estimator="mean",
            errorbar=("ci", 95),
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            legend.remove()
        if hline is not None:
            ax.axhline(hline, color="black", linestyle="--", linewidth=1)
        ax.set_title(str(method))
        ax.set_xlabel("Deployment round")
        ax.set_ylabel("Coverage")
    for ax in axes.flat[len(methods) :]:
        ax.set_visible(False)
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
        rect = [0.0, 0.0, 1.0, 0.93]
    else:
        rect = [0.0, 0.0, 1.0, 0.97]
    fig.suptitle(title, y=0.995)
    target_dir = figure_dir or FIGURES_MAIN_DIR
    pdf_path = ensure_parent(target_dir / f"{stem}.pdf")
    png_path = ensure_parent(target_dir / f"{stem}.png")
    fig.tight_layout(rect=rect)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    return pdf_path, png_path


def plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    stem: str,
    title: str,
    hue: str = "method",
    figure_dir: Path | None = None,
) -> tuple[Path, Path]:
    plot_frame = _display_method_labels(frame, method_col=hue) if hue == "method" else frame
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=plot_frame, x=x, y=y, hue=hue, ax=ax)
    sns.regplot(data=plot_frame, x=x, y=y, scatter=False, ax=ax, color="black", ci=95)
    ax.set_title(title)
    target_dir = figure_dir or FIGURES_MAIN_DIR
    pdf_path = ensure_parent(target_dir / f"{stem}.pdf")
    png_path = ensure_parent(target_dir / f"{stem}.png")
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    return pdf_path, png_path


def fit_groupwise_linear_trends(frame: pd.DataFrame, x: str, y: str, group: str) -> pd.DataFrame:
    rows = []
    for group_value, subset in frame.groupby(group, dropna=False):
        clean = subset[[x, y]].dropna()
        n_samples = int(len(clean))
        if n_samples >= 3 and clean[x].nunique() > 1:
            fit = stats.linregress(clean[x], clean[y])
            slope = float(fit.slope)
            intercept = float(fit.intercept)
            r_squared = float(fit.rvalue**2)
            pvalue = float(fit.pvalue)
            stderr = float(fit.stderr) if fit.stderr is not None else float("nan")
            dof = max(n_samples - 2, 1)
            critical = float(stats.t.ppf(0.975, dof)) if n_samples > 2 else float("nan")
            margin = critical * stderr if np.isfinite(stderr) else float("nan")
            slope_ci_low = slope - margin if np.isfinite(margin) else float("nan")
            slope_ci_high = slope + margin if np.isfinite(margin) else float("nan")
        else:
            slope = intercept = r_squared = pvalue = slope_ci_low = slope_ci_high = float("nan")
        rows.append(
            {
                group: group_value,
                "slope": slope,
                "slope_ci_low": slope_ci_low,
                "slope_ci_high": slope_ci_high,
                "r_squared": r_squared,
                "intercept": intercept,
                "pvalue": pvalue,
                "n_samples": n_samples,
            }
        )
    return pd.DataFrame(rows)


def plot_conditional_mean_curves(
    frame: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    stem: str,
    title: str,
    figure_dir: Path | None = None,
    n_bins: int = 6,
    x_label: str | None = None,
    y_label: str | None = None,
) -> tuple[Path, Path]:
    rows = []
    for group_value, subset in frame.groupby(group, dropna=False):
        clean = subset[[x, y]].dropna().copy()
        if clean.empty or clean[x].nunique() < 2:
            if not clean.empty:
                fallback = clean.rename(columns={x: "x_center", y: "y_mean"}).copy()
                fallback[group] = group_value
                rows.append(fallback.loc[:, ["x_center", "y_mean", group]])
            continue
        try:
            bins = pd.qcut(clean[x], q=min(n_bins, clean[x].nunique()), duplicates="drop")
        except ValueError:
            fallback = clean.rename(columns={x: "x_center", y: "y_mean"}).copy()
            fallback[group] = group_value
            rows.append(fallback.loc[:, ["x_center", "y_mean", group]])
            continue
        clean["_bin"] = bins
        grouped = clean.groupby("_bin", observed=False).agg(x_center=(x, "mean"), y_mean=(y, "mean")).reset_index(drop=True)
        grouped[group] = group_value
        rows.append(grouped)
    if not rows:
        raise ValueError("plot_conditional_mean_curves requires at least one non-degenerate group.")
    plot_frame = pd.concat(rows, ignore_index=True)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.lineplot(data=plot_frame, x="x_center", y="y_mean", hue=group, marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)
    target_dir = figure_dir or FIGURES_MAIN_DIR
    pdf_path = ensure_parent(target_dir / f"{stem}.pdf")
    png_path = ensure_parent(target_dir / f"{stem}.png")
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    return pdf_path, png_path


def plot_metric_panels(
    frame: pd.DataFrame,
    x: str,
    metrics: list[str] | tuple[str, ...],
    stem: str,
    title: str,
    hue: str | None = "method",
    figure_dir: Path | None = None,
    kind: str = "line",
    estimator: str | None = "mean",
    errorbar: tuple[str, int] | None = ("ci", 95),
    hlines: dict[str, float] | None = None,
    ylabels: dict[str, str] | None = None,
    metric_titles: dict[str, str] | None = None,
    x_label: str | None = None,
) -> tuple[Path, Path]:
    plot_frame = _display_method_labels(frame, method_col=hue) if hue == "method" else frame
    metrics = list(metrics)
    if not metrics:
        raise ValueError("plot_metric_panels requires at least one metric.")
    cols = 2 if len(metrics) > 1 else 1
    rows = int(math.ceil(len(metrics) / float(cols)))
    sns.set_theme(style="white")
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 3.8 * rows), squeeze=False)
    handles = labels = None
    for ax, metric in zip(axes.flat, metrics):
        if kind == "line":
            sns.lineplot(data=plot_frame, x=x, y=metric, hue=hue, ax=ax, estimator=estimator, errorbar=errorbar)
        elif kind == "bar":
            sns.barplot(data=plot_frame, x=x, y=metric, hue=hue, ax=ax)
        elif kind == "point":
            sns.pointplot(data=plot_frame, x=x, y=metric, hue=hue, ax=ax, errorbar=errorbar)
        else:
            raise ValueError(f"Unsupported panel plot kind: {kind}")
        legend = ax.get_legend()
        if legend is not None:
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            legend.remove()
        if hlines and metric in hlines:
            ax.axhline(hlines[metric], color="black", linestyle="--", linewidth=1)
        ax.set_title(metric_titles.get(metric, metric) if metric_titles else metric)
        ax.set_xlabel(x_label or x)
        ax.set_ylabel(ylabels.get(metric, metric) if ylabels else metric)
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        rect = [0.0, 0.0, 1.0, 0.89]
    else:
        rect = [0.0, 0.0, 1.0, 0.96]
    fig.suptitle(title, y=0.985)
    target_dir = figure_dir or FIGURES_MAIN_DIR
    pdf_path = ensure_parent(target_dir / f"{stem}.pdf")
    png_path = ensure_parent(target_dir / f"{stem}.png")
    fig.tight_layout(rect=rect)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    return pdf_path, png_path


def plot_distribution(
    frame: pd.DataFrame,
    x: str,
    stem: str,
    title: str,
    hue: str = "method",
    figure_dir: Path | None = None,
    kind: str = "kde",
) -> tuple[Path, Path]:
    plot_frame = _display_method_labels(frame, method_col=hue) if hue == "method" else frame
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 5))
    if kind == "kde":
        sns.kdeplot(data=plot_frame, x=x, hue=hue, ax=ax, common_norm=False, fill=False, warn_singular=False)
    elif kind == "hist":
        sns.histplot(data=plot_frame, x=x, hue=hue, ax=ax, stat="density", common_norm=False, element="step", fill=False)
    else:
        raise ValueError(f"Unsupported distribution plot kind: {kind}")
    ax.set_title(title)
    ax.set_xlabel(x)
    target_dir = figure_dir or FIGURES_MAIN_DIR
    pdf_path = ensure_parent(target_dir / f"{stem}.pdf")
    png_path = ensure_parent(target_dir / f"{stem}.png")
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    return pdf_path, png_path


def plot_heatmap(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    stem: str,
    title: str,
    figure_dir: Path | None = None,
) -> tuple[Path, Path]:
    pivot = frame.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.set_title(title)
    target_dir = figure_dir or FIGURES_MAIN_DIR
    pdf_path = ensure_parent(target_dir / f"{stem}.pdf")
    png_path = ensure_parent(target_dir / f"{stem}.png")
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    return pdf_path, png_path


def paired_comparison(
    frame: pd.DataFrame,
    metric: str,
    left: str,
    right: str,
    unit_cols: list[str] | None = None,
    method_col: str = "method",
) -> dict[str, float | str]:
    if metric not in frame.columns:
        return {"metric": metric, "left": left, "right": right, "test": "N/A", "pvalue": float("nan"), "n_pairs": 0}
    index_cols = unit_cols or [col for col in ["dataset", "seed", "round"] if col in frame.columns]
    if not index_cols:
        index_cols = [frame.index.name or "row_id"]
        frame = frame.reset_index(drop=False).rename(columns={"index": index_cols[0]})
    pivot = frame.pivot_table(index=index_cols, columns=method_col, values=metric, aggfunc="mean")
    if left not in pivot.columns or right not in pivot.columns:
        return {"metric": metric, "left": left, "right": right, "test": "N/A", "pvalue": float("nan"), "n_pairs": 0}
    paired = pivot[[left, right]].dropna()
    if len(paired) < 2:
        return {"metric": metric, "left": left, "right": right, "test": "N/A", "pvalue": float("nan"), "n_pairs": len(paired)}
    diff = paired[left] - paired[right]
    if np.allclose(diff.to_numpy(dtype=float), 0.0):
        stat, pvalue, test = 0.0, 1.0, "degenerate"
    elif abs(float(stats.skew(diff, bias=False))) < 1.0:
        stat, pvalue = stats.ttest_rel(paired[left], paired[right])
        test = "paired_t"
    else:
        stat, pvalue = stats.wilcoxon(paired[left], paired[right], zero_method="zsplit")
        test = "wilcoxon"
    return {
        "metric": metric,
        "left": left,
        "right": right,
        "test": test,
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "n_pairs": int(len(paired)),
        "left_mean": float(paired[left].mean()),
        "right_mean": float(paired[right].mean()),
        "effect_mean": float(diff.mean()),
    }


def pairwise_method_tests(
    frame: pd.DataFrame,
    metrics: list[str],
    reference_method: str = "PCRC",
    comparison_methods: list[str] | None = None,
    unit_cols: list[str] | None = None,
    method_col: str = "method",
) -> pd.DataFrame:
    methods = comparison_methods or [method for method in sorted(frame[method_col].dropna().unique()) if method != reference_method]
    rows = []
    for metric in metrics:
        for method in methods:
            rows.append(
                paired_comparison(
                    frame=frame,
                    metric=metric,
                    left=reference_method,
                    right=method,
                    unit_cols=unit_cols,
                    method_col=method_col,
                )
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    valid = result["pvalue"].notna()
    result["pvalue_bh"] = float("nan")
    if valid.any():
        result.loc[valid, "pvalue_bh"] = bh_correction(result.loc[valid, "pvalue"].tolist())
    result["significant_05_bh"] = result["pvalue_bh"] <= 0.05
    return result.sort_values(["metric", "right"]).reset_index(drop=True)


def bh_correction(pvalues: list[float]) -> list[float]:
    arr = pd.Series(pvalues, dtype=float)
    if arr.empty:
        return []
    order = np.argsort(arr.to_numpy(dtype=float))
    sorted_values = arr.iloc[order].to_numpy(dtype=float)
    adjusted = np.empty_like(sorted_values)
    m = len(sorted_values)
    running = 1.0
    for idx in range(m - 1, -1, -1):
        candidate = sorted_values[idx] * m / float(idx + 1)
        running = min(running, candidate)
        adjusted[idx] = min(running, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored.tolist()


def write_results_tex(title: str, frame: pd.DataFrame, path: Path) -> Path:
    section = "\n".join(
        [
            f"% BEGIN {title}",
            f"% {title}",
            frame.to_latex(index=False, escape=False),
            f"% END {title}",
        ]
    )
    target = ensure_parent(path)
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    legacy_pattern = re.compile(rf"(?ms)^% {re.escape(title)}\n\\begin{{tabular}}.*?\\end{{tabular}}\n?")
    existing = re.sub(legacy_pattern, "", existing)
    begin = f"% BEGIN {title}"
    end = f"% END {title}"
    if begin in existing and end in existing:
        start_idx = existing.index(begin)
        end_idx = existing.index(end) + len(end)
        prefix = existing[:start_idx].rstrip()
        suffix = existing[end_idx:].lstrip()
        updated = prefix + "\n\n" + section + "\n"
        if suffix:
            updated += "\n" + suffix
    else:
        updated = existing.rstrip()
        if updated:
            updated += "\n\n"
        updated += section + "\n"
    target.write_text(updated, encoding="utf-8")
    return path
