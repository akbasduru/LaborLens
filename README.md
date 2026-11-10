# LaborLens

**An AI Nowcasting Tool for Unemployment Using Real-Time Labor Market and Macroeconomic Data**

The Bureau of Labor Statistics (BLS) publishes each month's U.S. unemployment rate a few weeks after the month ends. LaborLens estimates the current month's rate before that release ("nowcasting"). It uses indicators that come out earlier: weekly jobless claims, job postings, consumer sentiment, inflation and interest rates.

The main model will be a deep neural network. It will be judged against two simple baselines. An AI model is only useful if it beats them.

## Research question

Can a deep neural network trained on real-time labor market and macroeconomic indicators estimate the current U.S. unemployment rate more accurately than a naive forecast and a standard ARIMA model? In particular, does it do better around sharp turning points such as the 2020 shock?

## Project status

| Stage | Status |
|---|---|
| Collect BLS data (unemployment rate, payrolls, JOLTS job openings) | Done |
| Collect FRED data (Fed funds rate, CPI, initial jobless claims, consumer sentiment) | Done |
| Collect Indeed Hiring Lab job postings index | Done |
| Naive and ARIMA(1,1,1) baselines | Done |
| Extend data back to 1995 so it covers the 2001, 2008 and 2020 recessions | Planned |
| Convert daily and weekly series to monthly values | Planned |
| Deep neural network nowcaster (MLP, LSTM, LSTNet) compared with the baselines | Planned |

## Baseline results

Both baselines are evaluated the same way. For each month in the test window, the model sees only earlier months and predicts the next one (an expanding window with one-step-ahead forecasts). Errors are in percentage points of the unemployment rate. Lower is better.

| Test window | Naive RMSE | Naive MAE | ARIMA RMSE | ARIMA MAE |
|---|---|---|---|---|
| 12 months | 0.085 | 0.073 | 0.110 | 0.094 |
| **24 months (primary)** | **0.104** | **0.083** | **0.118** | **0.098** |
| 36 months | 0.113 | 0.089 | 0.133 | 0.107 |

The naive forecast ("next month equals the last published value") beats ARIMA in every window. That sets a high bar for the neural network.

BLS did not publish an October 2025 value because of the government shutdown. That month is kept as missing. It is not filled in, and it is not scored.

## Data sources

| Source | Series | Frequency | Script |
|---|---|---|---|
| BLS | Unemployment rate, seasonally adjusted (`LNS14000000`) | Monthly | `src/unemployment_fetch.py` |
| BLS | Nonfarm payrolls (`CES0000000001`), JOLTS job openings (`JTU110099000000000HIL`) | Monthly | `src/apidata.py` |
| FRED | Federal funds rate (`FEDFUNDS`) | Monthly | `src/fred_data.py` |
| FRED | Consumer Price Index (`CPIAUCSL`) | Monthly | `src/fred_data.py` |
| FRED | Initial jobless claims (`ICSA`) | Weekly | `src/fred_data.py` |
| FRED | University of Michigan consumer sentiment (`UMCSENT`) | Monthly | `src/fred_data.py` |
| Indeed Hiring Lab | Job postings index | Daily | Downloaded manually to `data/raw/job-postings-headline-index.csv` |

## Repository layout

```
LaborLens/
├── data/
│   ├── raw/                  # BLS downloads and the Indeed postings index
│   └── *.csv                 # FRED series (fed_funds, cpi, claims, sentiment)
├── src/
│   ├── unemployment_fetch.py # BLS unemployment rate, 2015–2026
│   ├── apidata.py            # BLS payrolls, unemployment, JOLTS
│   ├── fred_data.py          # FRED indicators
│   ├── naive_baseline.py     # Naive baseline
│   └── arima_baseline.py     # ARIMA baseline
├── results/                  # Baseline scores, forecasts and plots
├── old results/              # Earlier ARIMA runs, kept for reference
├── presentation/             # Slide deck and pipeline diagram
└── requirements.txt
```

## Setup

This project needs Python 3 and the packages listed in `requirements.txt`.

### Option 1: uv (recommended)

Install [uv](https://docs.astral.sh/uv/) if you don't have it (`brew install uv` on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`). Then, from the repository root:

```bash
uv venv
uv pip install -r requirements.txt
```

### Option 2: pip

```bash
python3 -m pip install -r requirements.txt
```

### BLS API key

The BLS fetch scripts read your key from the `BLS_API_KEY` environment variable. You can get a free key from the [BLS registration page](https://data.bls.gov/registrationEngine/). Set it in your shell before running those scripts:

```bash
export BLS_API_KEY="your-key-here"
```

To keep it set in new terminal windows, add that line to `~/.zshrc`. The FRED script does not need a key.

## Usage

Run every script from the repository root so paths such as `data/` resolve correctly. With pip, replace `uv run python` with `python3`.

1. Fetch the data. The CSVs are already in `data/`, so this step is only needed to refresh them.

   ```bash
   uv run python src/unemployment_fetch.py
   uv run python src/fred_data.py
   ```

2. Run the baselines:

   ```bash
   uv run python src/naive_baseline.py
   uv run python src/arima_baseline.py
   ```

   Each script prints RMSE and MAE for the 12-, 24- and 36-month test windows. It also writes the following to `results/`:
   - `*_scores.csv`: errors for each test window
   - `*_forecasts.csv`: actual and predicted values for the 24-month window
   - `*_plot.png`: a chart of actual and predicted values

Settings such as the ARIMA order, the test windows and the minimum training length are listed at the top of each baseline script.

## Planned model

- **Framework:** PyTorch, with pandas, NumPy and scikit-learn for data preparation
- **Architectures to compare:** MLP, LSTM and LSTNet
- **Tuning:** Optuna
- **Interpretability:** Captum or SHAP, to show which indicators drive each estimate
- **Evaluation:** the same expanding-window setup and error metrics as the baselines, with extra attention to the months around 2020

## Related work

- Xu & Liu (2025). *Labor Market Employment Rate Prediction and Optimization Based on Deep Neural Networks.* ICCSIE 2025, ACM, pp. 101–106. [doi:10.1145/3759179.3759194](https://doi.org/10.1145/3759179.3759194)
- Abbouchi et al. (2025). *Forecasting Labor Markets with LSTNet.* [arXiv:2507.01979](https://arxiv.org/abs/2507.01979)

## Author

Duru Akbas, Allegheny College
