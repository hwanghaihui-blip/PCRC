# Package Manifest

## Included

- `src/pcrc/`: PCRC implementation, simulators, data loaders, data transformations, experiment runners, reporting, and contract metadata.
- `configs/*.yaml`: active experiment and smoke configurations.
- `scripts/download_datasets.py`: downloads the public datasets used by packaged experiments.
- `scripts/prepare_data.py`: materializes lightweight processed data outputs and validates the data pipelines.
- `scripts/run_*.py`: maintained PCRC-only experiment entry points for D0, D1/M5, D2/credit, off-policy, and smoke runs.
- `tests/`: focused package tests that match the included source and scripts.
- Empty output directories for `data`, `rollouts`, `figures`, `tables`, `reports`, and `audit/runtime`.
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, and this README/manifest.

## Excluded

- `data/raw/`: large raw public datasets. Download them with `scripts/download_datasets.py`.
- `data/processed/`: generated intermediate data. Rebuild with experiments or `scripts/prepare_data.py`.
- `rollouts/`, `figures/`, `tables/`, `reports/`, `audit/`: generated outputs and historical audit records.
- `deliverables/`: manuscript DOCX/PDF files, historical zip bundles, figure mini-data bundles, and submission artifacts.
- Manuscript-only scripts that depend on historical submission directories or DOCX templates.
- External comparator repositories, comparator adapters, and comparator run code.
- Auxiliary method-variant study source, configs, scripts, and outputs.
- `__pycache__`, `.pytest_cache`, `.git`, and other local cache/metadata files.

## Comparator Handling

The packaged source is PCRC-only. It includes no external comparator implementations and no in-repository comparator adapters; all active configs instantiate `PCRC` only.
