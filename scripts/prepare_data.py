#!/usr/bin/env python3
"""Download and prepare lightweight PCRC data artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from pcrc.constants import PROCESSED_DATA_DIR, RAW_DATA_DIR
from pcrc.data.credit import download_credit_dataset, load_credit_frame
from pcrc.data.m5 import (
    M5ResponseKernel,
    build_m5_subset_panel,
    build_subset_manifest,
    download_m5_dataset,
)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def prepare_credit() -> dict[str, str | int]:
    archive = download_credit_dataset()
    frame = load_credit_frame(archive)
    schema = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing": [int(frame[column].isna().sum()) for column in frame.columns],
        }
    )
    _write_frame(schema, PROCESSED_DATA_DIR / "credit_schema.csv")
    _write_frame(frame.head(256), PROCESSED_DATA_DIR / "credit_sample.parquet")
    return {
        "archive": str(archive),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "schema": str(PROCESSED_DATA_DIR / "credit_schema.csv"),
        "sample": str(PROCESSED_DATA_DIR / "credit_sample.parquet"),
    }


def prepare_m5(token: str | None, max_series: int) -> dict[str, str | int]:
    paths = download_m5_dataset(token=token)
    panel = build_m5_subset_panel(
        RAW_DATA_DIR / "m5",
        states=["CA", "TX"],
        categories=["FOODS", "HOUSEHOLD"],
        stores=["CA_1", "CA_2", "TX_1", "TX_2"],
        max_rows=max_series,
    )
    kernel = M5ResponseKernel()
    panel = kernel.annotate_anchor_loss(panel)
    manifest = build_subset_manifest(panel)
    elasticity = kernel.fit_elasticity(panel)
    _write_frame(manifest, PROCESSED_DATA_DIR / "m5_subset_manifest.csv")
    _write_frame(panel.head(2048), PROCESSED_DATA_DIR / "m5_panel_sample.parquet")
    _write_frame(elasticity.head(2048), PROCESSED_DATA_DIR / "m5_elasticity_sample.parquet")
    return {
        "files": json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True),
        "panel_rows": int(len(panel)),
        "manifest": str(PROCESSED_DATA_DIR / "m5_subset_manifest.csv"),
        "panel_sample": str(PROCESSED_DATA_DIR / "m5_panel_sample.parquet"),
        "elasticity_sample": str(PROCESSED_DATA_DIR / "m5_elasticity_sample.parquet"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "m5", "credit"], default="all")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--m5-max-series", type=int, default=300)
    args = parser.parse_args()

    token = os.environ.get(args.hf_token_env)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    payload: dict[str, dict[str, str | int]] = {}
    if args.dataset in {"all", "credit"}:
        payload["credit"] = prepare_credit()
    if args.dataset in {"all", "m5"}:
        payload["m5"] = prepare_m5(token=token, max_series=args.m5_max_series)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
