# PCRC Standalone Code Package

This package contains the code needed to run the PCRC experiments on a local machine. 

The package keeps the project runnable and compact:

- `src/pcrc/`: PCRC implementation, simulators, data-processing code, reporting utilities.
- `scripts/`: command-line entry points for downloading/preparing data and running experiments.
- `configs/`: experiment configurations, including a small CPU smoke config.
- `tests/`: focused tests for core methods, schemas, data loaders, and smoke experiments.
- `data/`, `rollouts/`, `figures/`, `tables/`, `reports/`, `audit/`: empty output directories created for local runs.


## Environment

Python 3.10 or newer is required. A clean virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If editable installs are inconvenient, use:

```bash
python -m pip install -r requirements-dev.txt
export PYTHONPATH=src
```

## Quick Smoke Run

Run a small synthetic experiment that needs no external data:

```bash
PYTHONPATH=src python scripts/run_d0_experiment.py --config configs/smoke_cpu.yaml
```

Run the focused test subset:

```bash
PYTHONPATH=src pytest tests/test_methods.py tests/test_schema.py tests/test_d0.py
```

The broader tests and D1/D2 experiments download public datasets on first use.

## Data Download And Preparation

D0 is synthetic and generated in code. D1 and D2 require public datasets:

- D1 M5: Hugging Face dataset `denephew/M5_Forecasting`, stored under `data/raw/m5/`.
- D2 Credit: UCI Default of Credit Card Clients zip, stored under `data/raw/credit/`.

Download all required public datasets:

```bash
HF_TOKEN=your_token_if_needed PYTHONPATH=src python scripts/download_datasets.py
```

Prepare lightweight processed artifacts that verify the data-processing pipeline:

```bash
HF_TOKEN=your_token_if_needed PYTHONPATH=src python scripts/prepare_data.py --dataset all
```

The data-processing logic lives in:

- `src/pcrc/data/m5.py`
- `src/pcrc/data/credit.py`

## Main Experiment Commands

```bash
PYTHONPATH=src python scripts/run_d0_experiment.py --config configs/exp2_phase_transition.yaml
HF_TOKEN=your_token_if_needed PYTHONPATH=src python scripts/run_d1_m5.py --config configs/exp3_m5_case.yaml
PYTHONPATH=src python scripts/run_d2_credit.py --config configs/exp4_credit_case.yaml
PYTHONPATH=src python scripts/run_offpolicy_d0.py --config configs/exp5_offpolicy.yaml
```



See `PACKAGE_MANIFEST.md` for a detailed include/exclude list.
