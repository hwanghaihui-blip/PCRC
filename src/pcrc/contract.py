"""Experiment contract for the PCRC-only standalone package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pcrc.constants import AUDIT_DIR, PRIMARY_METHOD, REPORTS_DIR
from pcrc.utils import dump_json, ensure_parent


CONTRACT_VERSION = "2026-06-24-pcrc-only"
SOURCE_SCOPE_RULE = (
    "This standalone package exposes only the paper's proposed method, PCRC, "
    "for the packaged main experiments. External comparator and auxiliary method-variant implementations are not bundled."
)


@dataclass(frozen=True, slots=True)
class MethodSource:
    method: str
    family: str
    package_scope: str
    implementation_status: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    experiment: str
    dataset: str
    artifact_type: str
    number: str
    stem: str
    title: str
    section: str


@dataclass(frozen=True, slots=True)
class DatasetSource:
    dataset: str
    role: str
    source: str
    download_entry: str
    local_cache: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class TuningBoundary:
    experiment: str
    scenario_scope: str
    tunable_parameters: str
    locked_parameters: str
    escalation_rule: str


@dataclass(frozen=True, slots=True)
class CoverageRule:
    experiment: str
    target_expression: str
    mean_tolerance: float
    ci_low_tolerance: float
    tie_breakers: str
    failure_rule: str


METHOD_SOURCES: tuple[MethodSource, ...] = (
    MethodSource(
        method=PRIMARY_METHOD,
        family="pcrc",
        package_scope="proposed_method",
        implementation_status="native implementation included in src/pcrc/methods/conformal.py",
        notes="All packaged experiment configs instantiate PCRC only.",
    ),
)


DATASET_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        dataset="D0 synthetic",
        role="main theorem validation",
        source="this repository synthetic closed-loop simulator",
        download_entry="generated in-code via scripts/run_d0_experiment.py",
        local_cache="src/pcrc/simulators/d0.py",
        notes="Stable, critical, and unstable regimes for coverage collapse, fixed-point, and gap-regret studies.",
    ),
    DatasetSource(
        dataset="D1 M5",
        role="main case study",
        source="Hugging Face dataset denephew/M5_Forecasting",
        download_entry="scripts/download_datasets.py / pcrc.data.m5.download_m5_dataset",
        local_cache="data/raw/m5",
        notes="Semi-synthetic price-response environment built on the M5 hierarchy.",
    ),
    DatasetSource(
        dataset="D2 Credit",
        role="main case study",
        source="UCI Default of Credit Card Clients",
        download_entry="scripts/download_datasets.py / pcrc.data.credit.download_credit_dataset",
        local_cache="data/raw/credit/default_of_credit_card_clients.zip",
        notes="Approval-threshold closed-loop credit-risk environment with approval-floor auditing.",
    ),
)


ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("exp1_overview", "D0+D1+D2", "figure", "fig1_1", "pcrc_closed_loop_overview", "Figure 1-1: PCRC closed-loop system overview", "Overview"),
    ArtifactSpec("exp1_overview", "D0+D1+D2", "table", "table1_1", "method_sources", "Table 1-1: PCRC method source", "Overview"),
    ArtifactSpec("exp1_overview", "D0+D1+D2", "table", "table1_2", "dataset_sources", "Table 1-2: Datasets and download entry points", "Overview"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_1", "pre_post_coverage", "Figure 2-1: Pre vs post-decision coverage by round", "D0 main evidence"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_2", "set_volume", "Figure 2-2: Prediction-set volume by round", "D0 main evidence"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_3", "operator_curve", "Figure 2-3: Performative quantile operator", "D0 operator/root"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_4", "tau_trajectory", "Figure 2-4: Tau trajectories from different initializations", "D0 operator/root"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_5", "fp_residual", "Figure 2-5: Fixed-point residual by round", "D0 operator/root"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_6", "phase_heatmap", "Figure 2-6: Stability heatmap", "D0 stability"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_7", "phase_portrait", "Figure 2-7: Critical-regime phase portrait", "D0 stability"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_8", "margin_speed", "Figure 2-8: Convergence speed vs stability margin", "D0 stability"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_9", "gap_regret_scatter", "Figure 2-9: Coverage gap to regret", "D0 main evidence"),
    ArtifactSpec("exp2_phase_transition", "D0", "figure", "fig2_10", "gap_regret_regime", "Figure 2-10: Coverage-gap to regret conditional means by regime", "D0 main evidence"),
    ArtifactSpec("exp2_phase_transition", "D0", "table", "table2_1", "coverage_collapse_summary", "Table 2-1: D0 PCRC overall summary", "D0 main evidence"),
    ArtifactSpec("exp2_phase_transition", "D0", "table", "table2_2", "fixed_point_convergence", "Table 2-2: D0 fixed-point convergence summary", "D0 operator/root"),
    ArtifactSpec("exp2_phase_transition", "D0", "table", "table2_3", "gap_regret_pairs", "Table 2-3: D0 gap-regret slope fit summary", "D0 main evidence"),
    ArtifactSpec("exp3_m5_case", "D1", "figure", "fig3_1", "m5_subset_layout", "Figure 3-1: M5 stratified subset composition", "D1 case study"),
    ArtifactSpec("exp3_m5_case", "D1", "figure", "fig3_2", "m5_coverage_compare", "Figure 3-2: M5 pre/post coverage comparison", "D1 case study"),
    ArtifactSpec("exp3_m5_case", "D1", "figure", "fig3_3", "m5_profit_cvar_frontier", "Figure 3-3: M5 profit-CVaR frontier", "D1 case study"),
    ArtifactSpec("exp3_m5_case", "D1", "figure", "fig3_4", "m5_elasticity_groups", "Figure 3-4: M5 elasticity-group coverage/regret comparison", "D1 slices"),
    ArtifactSpec("exp3_m5_case", "D1", "figure", "fig3_5", "m5_high_elasticity_trajectory", "Figure 3-5: High-elasticity representative trajectory", "D1 trajectories"),
    ArtifactSpec("exp3_m5_case", "D1", "figure", "fig3_6", "m5_low_elasticity_trajectory", "Figure 3-6: Low-elasticity representative trajectory", "D1 trajectories"),
    ArtifactSpec("exp3_m5_case", "D1", "table", "table3_1", "m5_overall_summary", "Table 3-1: D1 PCRC overall summary", "D1 case study"),
    ArtifactSpec("exp3_m5_case", "D1", "table", "table3_2", "m5_slices", "Table 3-2: D1 slice summary", "D1 slices"),
    ArtifactSpec("exp4_credit_case", "D2", "figure", "fig4_1", "threshold_trajectory", "Figure 4-1: Threshold evolution", "D2 case study"),
    ArtifactSpec("exp4_credit_case", "D2", "figure", "fig4_2", "approval_cvar_frontier", "Figure 4-2: Approval-CVaR frontier", "D2 case study"),
    ArtifactSpec("exp4_credit_case", "D2", "figure", "fig4_3", "near_threshold_coverage", "Figure 4-3: Near-threshold post-decision coverage", "D2 case study"),
    ArtifactSpec("exp4_credit_case", "D2", "figure", "fig4_4", "tail_loss_distribution", "Figure 4-4: Portfolio tail-loss distribution", "D2 case study"),
    ArtifactSpec("exp4_credit_case", "D2", "figure", "fig4_5", "pool_composition", "Figure 4-5: Applicant-pool composition shift", "D2 case study"),
    ArtifactSpec("exp4_credit_case", "D2", "table", "table4_1", "credit_overall_summary", "Table 4-1: D2 PCRC overall summary", "D2 case study"),
    ArtifactSpec("exp4_credit_case", "D2", "table", "table4_2", "credit_slices", "Table 4-2: D2 slice summary", "D2 case study"),
    ArtifactSpec("exp5_offpolicy", "D0+D2", "figure", "fig5_1", "offpolicy_coverage", "Figure 5-1: PCRC off-policy post-decision coverage", "Experiment 5"),
    ArtifactSpec("exp5_offpolicy", "D0+D2", "figure", "fig5_2", "ess_vs_error", "Figure 5-2: ESS vs coverage error", "Experiment 5"),
    ArtifactSpec("exp5_offpolicy", "D0+D2", "figure", "fig5_3", "weight_distribution", "Figure 5-3: Off-policy weight diagnostics", "Experiment 5"),
    ArtifactSpec("exp5_offpolicy", "D0+D2", "table", "table5_1", "offpolicy_summary", "Table 5-1: PCRC off-policy summary", "Experiment 5"),
)


COMMON_SUMMARY_TABLE_COLUMNS: tuple[str, ...] = (
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
)


TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "table1_1": ("method", "family", "package_scope", "implementation_status", "notes"),
    "table1_2": ("dataset", "role", "source", "download_entry", "local_cache", "notes"),
    "table2_1": COMMON_SUMMARY_TABLE_COLUMNS,
    "table2_2": (
        "regime",
        "method",
        "tau_init",
        "C_pre",
        "C_post",
        "gap",
        "set_size_or_volume",
        "risk(CVaR)",
        "regret",
        "fp_residual",
        "runtime_or_iteration_count",
        "tau_t_mean",
        "tau_t_ci_low",
        "tau_t_ci_high",
        "fp_residual_mean",
        "fp_residual_ci_low",
        "fp_residual_ci_high",
        "round_count",
    ),
    "table2_3": (
        "regime",
        "method",
        "C_pre",
        "C_post",
        "gap",
        "set_size_or_volume",
        "risk(CVaR)",
        "regret",
        "fp_residual",
        "runtime_or_iteration_count",
        "slope",
        "slope_ci_low",
        "slope_ci_high",
        "r_squared",
        "intercept",
        "pvalue",
        "n_samples",
    ),
    "table3_1": COMMON_SUMMARY_TABLE_COLUMNS,
    "table3_2": (
        "method",
        "elasticity_group",
        "state_id",
        "cat_id",
        "promo_flag",
        "event_flag",
        "C_pre",
        "C_post",
        "gap",
        "set_size_or_volume",
        "risk(CVaR)",
        "regret",
        "fp_residual",
        "runtime_or_iteration_count",
        "mean_profit",
        "set_volume",
        "cvar_proxy",
    ),
    "table4_1": COMMON_SUMMARY_TABLE_COLUMNS,
    "table4_2": (
        "method",
        "slice_name",
        "approval_rate",
        "C_pre",
        "C_post",
        "gap",
        "set_size_or_volume",
        "risk(CVaR)",
        "regret",
        "fp_residual",
        "runtime_or_iteration_count",
        "cvar_loss",
        "portfolio_loss",
    ),
    "table5_1": (
        "dataset",
        "method",
        "coverage",
        "coverage_ci_low",
        "coverage_ci_high",
        "gap",
        "regret",
        "regret_ci_low",
        "regret_ci_high",
        "variance",
        "ESS",
        "ESS_ci_low",
        "ESS_ci_high",
        "C_pre",
        "C_post",
        "set_size_or_volume",
        "risk(CVaR)",
        "fp_residual",
        "runtime_or_iteration_count",
        "coverage_gate",
        "coverage_priority_rank",
    ),
}


TUNING_BOUNDARIES: tuple[TuningBoundary, ...] = (
    TuningBoundary(
        experiment="exp2_phase_transition",
        scenario_scope="within-regime only",
        tunable_parameters="temperature, geometry_type, root_blend, adapt_rate, rolling_window",
        locked_parameters="action_grid, regime definitions, figure/table numbering, summary columns",
        escalation_rule="Tune only inside a fixed regime and record any deviation if the coverage gate still fails.",
    ),
    TuningBoundary(
        experiment="exp3_m5_case",
        scenario_scope="within fixed representative subset and deterministic deep-case deployment schedule",
        tunable_parameters="temperature, geometry_type, response-kernel nuisance scales, monte_carlo_draws, trajectory_points_per_level",
        locked_parameters="state/category/store subset, time split ordering, deterministic schedule recipe, response-kernel family, figure/table numbering",
        escalation_rule="Do not expand or swap the representative subset after packaging.",
    ),
    TuningBoundary(
        experiment="exp4_credit_case",
        scenario_scope="within pseudo-temporal batches under a fixed applicant pool definition",
        tunable_parameters="temperature, geometry_type, composition_strength, near_threshold_band",
        locked_parameters="approval floor, threshold grid, pseudo-temporal ordering rule, summary columns, figure/table numbering",
        escalation_rule="Retune only inside the locked pool generator.",
    ),
    TuningBoundary(
        experiment="exp5_offpolicy",
        scenario_scope="within overlap regime and weighting choices",
        tunable_parameters="overlap softness, temperature, geometry_type",
        locked_parameters="datasets covered, behavior-policy families, figure/table numbering",
        escalation_rule="Off-policy tuning cannot feed back into the mainline D0/D1/D2 configuration.",
    ),
)


COVERAGE_RULES: tuple[CoverageRule, ...] = (
    CoverageRule("exp2_phase_transition", "covered_post target = 1 - nominal_alpha", 0.01, 0.03, "Smaller mean shortfall, then lower risk/regret/set volume.", "Coverage gate dominates all other metrics."),
    CoverageRule("exp3_m5_case", "covered_post target = 1 - nominal_alpha", 0.01, 0.03, "Smaller mean shortfall, then lower regret, then higher mean profit.", "High/low elasticity slices inherit the same coverage-first gate."),
    CoverageRule("exp4_credit_case", "covered_post target = 1 - nominal_alpha", 0.01, 0.03, "Smaller mean shortfall, then smaller CVaR, then higher approval rate subject to the approval floor.", "Approval-rate improvements never outrank a coverage failure."),
    CoverageRule("exp5_offpolicy", "covered_post target = 1 - nominal_alpha", 0.02, 0.05, "Smaller coverage error, then larger ESS, then smaller weight instability.", "D0 and D2 must both be reported."),
)


def _artifact_number_order(number: str) -> tuple[int, int]:
    prefix, _, suffix = number.partition("_")
    head = "".join(ch for ch in prefix if ch.isdigit())
    tail = "".join(ch for ch in suffix if ch.isdigit())
    return (int(head or 0), int(tail or 0))


def method_sources_frame() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in METHOD_SOURCES)


def method_label_map() -> dict[str, str]:
    return {PRIMARY_METHOD: PRIMARY_METHOD}


def canonical_method_rosters() -> dict[str, list[str]]:
    return {
        "mainline": [PRIMARY_METHOD],
        "offpolicy": [PRIMARY_METHOD],
    }


def dataset_sources_frame() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in DATASET_SOURCES)


def artifact_manifest_frame() -> pd.DataFrame:
    frame = pd.DataFrame(asdict(item) for item in ARTIFACT_SPECS)
    frame["_number_order"] = frame["number"].map(_artifact_number_order)
    return frame.sort_values(["experiment", "artifact_type", "_number_order"]).drop(columns=["_number_order"]).reset_index(drop=True)


def tuning_boundaries_frame() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in TUNING_BOUNDARIES)


def coverage_rules_frame() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in COVERAGE_RULES)


def table_schema_frame() -> pd.DataFrame:
    rows = [{"table_number": number, "columns": ", ".join(columns)} for number, columns in TABLE_SCHEMAS.items()]
    return pd.DataFrame(rows).sort_values("table_number").reset_index(drop=True)


def table_columns(number: str) -> tuple[str, ...] | None:
    return TABLE_SCHEMAS.get(number)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row.tolist()) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, sep, *rows])


def _write_text(path: Path, content: str) -> Path:
    target = ensure_parent(path)
    target.write_text(content, encoding="utf-8")
    return target


def write_contract_bundle() -> dict[str, Path]:
    method_frame = method_sources_frame()
    dataset_frame = dataset_sources_frame()
    artifact_frame = artifact_manifest_frame()
    schema_frame = table_schema_frame()
    tuning_frame = tuning_boundaries_frame()
    coverage_frame = coverage_rules_frame()
    contract_payload = {
        "contract_version": CONTRACT_VERSION,
        "source_scope_rule": SOURCE_SCOPE_RULE,
        "method_sources": method_frame.to_dict(orient="records"),
        "dataset_sources": dataset_frame.to_dict(orient="records"),
        "artifact_manifest": artifact_frame.to_dict(orient="records"),
        "table_schemas": schema_frame.to_dict(orient="records"),
        "tuning_boundaries": tuning_frame.to_dict(orient="records"),
        "coverage_rules": coverage_frame.to_dict(orient="records"),
    }
    json_path = AUDIT_DIR / "runtime" / "experiment_contract.json"
    dump_json(json_path, contract_payload)

    experiment_manifest_md = ["# Experiment Manifest", "", f"Contract version: `{CONTRACT_VERSION}`.", ""]
    for experiment, group in artifact_frame.groupby("experiment", sort=False):
        dataset = group["dataset"].iloc[0]
        experiment_manifest_md.append(f"## {experiment} ({dataset})")
        experiment_manifest_md.append("")
        experiment_manifest_md.append(_frame_to_markdown(group[["artifact_type", "number", "stem", "title", "section"]]))
        experiment_manifest_md.append("")

    contract_md = [
        "# Experiment Contract",
        "",
        f"Contract version: `{CONTRACT_VERSION}`.",
        "",
        "## Package Scope",
        "",
        SOURCE_SCOPE_RULE,
        "",
        "## Method Source",
        "",
        _frame_to_markdown(method_frame),
        "",
        "## Dataset Sources",
        "",
        _frame_to_markdown(dataset_frame),
        "",
        "## Numbered Artifacts",
        "",
        _frame_to_markdown(artifact_frame),
        "",
        "## Frozen Table Schemas",
        "",
        _frame_to_markdown(schema_frame),
        "",
        "## Scenario-Internal Tuning Boundaries",
        "",
        _frame_to_markdown(tuning_frame),
        "",
        "## Coverage-First Pass Criteria",
        "",
        _frame_to_markdown(coverage_frame),
        "",
    ]

    return {
        "contract_json": json_path,
        "experiment_manifest": _write_text(REPORTS_DIR / "experiment_manifest.md", "\n".join(experiment_manifest_md)),
        "experiment_contract": _write_text(REPORTS_DIR / "experiment_contract.md", "\n".join(contract_md)),
    }
