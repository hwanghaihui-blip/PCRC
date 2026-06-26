#!/usr/bin/env python3
"""Run a configured D0 experiment."""

from __future__ import annotations

import argparse

from pcrc.config import ExperimentConfig
from pcrc.experiments.d0 import run_d0_experiment
from pcrc.io import initialize_reports, write_audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp2_phase_transition.yaml")
    args = parser.parse_args()
    initialize_reports()
    cfg = ExperimentConfig.from_yaml(args.config)
    write_audit_manifest({"experiment": cfg.name, "dataset": cfg.dataset})
    outputs = run_d0_experiment(cfg)
    print(outputs.summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
