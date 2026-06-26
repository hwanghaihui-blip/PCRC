#!/usr/bin/env python3
"""Run off-policy D0 diagnostics."""

from __future__ import annotations

import argparse

from pcrc.config import ExperimentConfig
from pcrc.experiments.offpolicy import run_offpolicy_d0
from pcrc.io import initialize_reports, write_audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp5_offpolicy.yaml")
    args = parser.parse_args()
    initialize_reports()
    cfg = ExperimentConfig.from_yaml(args.config)
    write_audit_manifest({"experiment": cfg.name, "dataset": cfg.dataset})
    outputs = run_offpolicy_d0(cfg)
    print(outputs.summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
