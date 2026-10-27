# LaborLens
Detecting Structural Breaks in Labor Market Forecasting: A Regime-Switching Approach Using Real-Time Job Postings Data

## Setup

This project needs Python 3 and the packages listed in `requirements.txt`.

### Option 1: uv (recommended)

Install [uv](https://docs.astral.sh/uv/) if you don't have it (`brew install uv` on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`). Then, from the repository root:

```bash
uv venv
uv pip install -r requirements.txt
```

Run any program with `uv run`, which uses the `.venv` environment automatically:

```bash
uv run python src/arima_baseline.py
```

### Option 2: pip

```bash
python3 -m pip install -r requirements.txt
python3 src/arima_baseline.py
```

Run the scripts from the repository root so relative paths such as `data/` resolve correctly.
