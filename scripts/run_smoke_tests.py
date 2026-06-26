#!/usr/bin/env python3
"""Smoke-test the PCRC project."""

from __future__ import annotations

import json

from pcrc.config import ExperimentConfig
from pcrc.data.credit import download_credit_dataset, load_credit_frame
from pcrc.io import initialize_reports, write_audit_manifest
from pcrc.experiments.d0 import run_d0_experiment


def main() -> None:
    initialize_reports()
    write_audit_manifest({"smoke_test": True})
    cfg = ExperimentConfig(
        name="smoke_d0",
        dataset="D0",
        methods=["PCRC"],
        seeds=[7],
        rounds=20,
        calibration_budget=128,
        action_grid=[0.85, 0.925, 1.0, 1.075, 1.15],
        params={"regimes": ["S1_stable"], "pretrain_samples": 512, "temperature": 0.1},
    )
    outputs = run_d0_experiment(cfg)
    archive = download_credit_dataset()
    credit = load_credit_frame(archive)
    payload = {
        "d0_rows": len(outputs.rollout_frame),
        "d0_summary_rows": len(outputs.summary_frame),
        "credit_shape": list(credit.shape),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
