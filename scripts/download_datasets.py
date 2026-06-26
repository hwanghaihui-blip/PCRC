#!/usr/bin/env python3
"""Download external datasets used by PCRC."""

from __future__ import annotations

import argparse
import json
import os

from pcrc.data.credit import download_credit_dataset
from pcrc.data.m5 import download_m5_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.hf_token_env)
    payload = {
        "m5": {k: str(v) for k, v in download_m5_dataset(token=token).items()},
        "credit": str(download_credit_dataset()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
