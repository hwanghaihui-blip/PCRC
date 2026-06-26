#!/usr/bin/env python3
"""Run the D1 M5 semi-synthetic experiment."""

from __future__ import annotations

import argparse
import os

from pcrc.config import ExperimentConfig
from pcrc.experiments.d1_m5 import run_d1_m5_experiment
from pcrc.io import initialize_reports, write_audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp3_m5_case.yaml")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()
    initialize_reports()
    cfg = ExperimentConfig.from_yaml(args.config)
    token = os.environ.get(args.hf_token_env)
    write_audit_manifest({"experiment": cfg.name, "dataset": cfg.dataset})
    outputs = run_d1_m5_experiment(cfg, hf_token=token)
    print(outputs.summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
