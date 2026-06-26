"""Filesystem and audit helpers."""

from __future__ import annotations

from pathlib import Path

from pcrc.contract import CONTRACT_VERSION, write_contract_bundle
from pcrc.constants import AUDIT_DIR, FIGURES_APP_DIR, FIGURES_MAIN_DIR, FORMAL_EXPERIMENTS, REPORTS_DIR, TABLES_APP_DIR, TABLES_MAIN_DIR
from pcrc.overview import write_overview_artifacts
from pcrc.utils import dump_json, env_summary, git_hash


def write_audit_manifest(extra: dict | None = None) -> Path:
    payload = {
        "git_hash": git_hash(),
        "environment": env_summary(),
        "contract_version": CONTRACT_VERSION,
    }
    if extra:
        payload.update(extra)
    path = AUDIT_DIR / "runtime" / "manifest.json"
    dump_json(path, payload)
    return path


def write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def ensure_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(content, encoding="utf-8")
    return target


def initialize_reports() -> None:
    write_contract_bundle()
    write_overview_artifacts()
    ensure_text(
        REPORTS_DIR / "reproducibility_checklist.md",
        "# Reproducibility Checklist\n\n- Git hash\n- Environment versions\n- Seeds\n- Data sources\n",
    )
    ensure_text(
        REPORTS_DIR / "fatal_flaws_memo.md",
        "# Fatal Flaws Memo\n\nDocument divergence, instability, overlap failure, and misspecification cases here.\n",
    )
    ensure_text(REPORTS_DIR / "main_results.tex", "% Auto-generated main results.\n")
    ensure_text(REPORTS_DIR / "appendix_results.tex", "% Auto-generated appendix results.\n")


def is_formal_experiment(name: str) -> bool:
    return str(name) in FORMAL_EXPERIMENTS


def artifact_stem(name: str, stem: str) -> str:
    return stem if is_formal_experiment(name) else f"{name}_{stem}"


def figure_dir_for_run(name: str) -> Path:
    return FIGURES_MAIN_DIR if is_formal_experiment(name) else FIGURES_APP_DIR


def table_dir_for_run(name: str) -> Path:
    return TABLES_MAIN_DIR if is_formal_experiment(name) else TABLES_APP_DIR


def results_tex_path_for_run(name: str) -> Path:
    return REPORTS_DIR / ("main_results.tex" if is_formal_experiment(name) else "appendix_results.tex")
