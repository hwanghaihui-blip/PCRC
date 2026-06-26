"""Shared constants and filesystem defaults."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
ROLLOUTS_DIR = REPO_ROOT / "rollouts"
FIGURES_MAIN_DIR = REPO_ROOT / "figures" / "main"
FIGURES_APP_DIR = REPO_ROOT / "figures" / "app"
TABLES_MAIN_DIR = REPO_ROOT / "tables" / "main"
TABLES_APP_DIR = REPO_ROOT / "tables" / "app"
REPORTS_DIR = REPO_ROOT / "reports"
AUDIT_DIR = REPO_ROOT / "audit"
SEED_LIST_MAIN = [7, 11, 19, 23, 29]
SEED_LIST_HIGH_VAR = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
DEFAULT_DEVICE = "cuda:1"
NOMINAL_ALPHA = 0.1
PRIMARY_METHOD = "PCRC"
FORMAL_EXPERIMENTS = {
    "exp2_phase_transition",
    "exp3_m5_case",
    "exp4_credit_case",
    "exp5_offpolicy",
}
PCRC_METHODS = [PRIMARY_METHOD]
MAINLINE_METHODS = PCRC_METHODS
OFFPOLICY_METHODS = PCRC_METHODS
SMOKE_METHODS = PCRC_METHODS
CONFIG_METHOD_ROSTERS = {
    "exp2_phase_transition": MAINLINE_METHODS,
    "exp3_m5_case": MAINLINE_METHODS,
    "exp4_credit_case": MAINLINE_METHODS,
    "exp5_offpolicy": OFFPOLICY_METHODS,
    "smoke_d0": SMOKE_METHODS,
    "smoke_cpu": SMOKE_METHODS,
}
ROLLOUT_COLUMNS = [
    "dataset",
    "method",
    "seed",
    "round",
    "context_id",
    "action",
    "outcome",
    "predicted_center",
    "set_lower_or_summary",
    "set_upper_or_summary",
    "set_volume",
    "score",
    "tau",
    "covered_pre",
    "covered_post",
    "fp_residual",
    "risk_value",
    "regret",
    "is_on_policy",
    "propensity",
    "importance_weight",
    "ESS",
    "geometry_type",
    "temperature",
    "surrogate_misspec_level",
]
