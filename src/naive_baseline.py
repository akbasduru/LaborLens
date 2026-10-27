"""
Naive baseline for the US unemployment rate (BLS series LNS14000000).

Input : data/raw/unemployment_2015_2026.csv, produced by your BLS fetch script
        (columns: series_id, date, year, period, period_name, value)
Method: one-step-ahead naive forecasts. The forecast for month t is simply the
        last published value before t. Nothing is fitted or tuned.
        Uses the same data, test windows and scoring as arima_baseline.py.
Output: results/naive_baseline_scores.csv, naive_baseline_forecasts.csv,
        naive_baseline_plot.png (RMSE / MAE per test window, plus the primary window)

Run:  uv run python src/naive_baseline.py   (from the project root)
Needs: uv pip install -r requirements.txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings (keep TEST_SIZES etc. identical to arima_baseline.py)
# ----------------------------------------------------------------------------
# Project root = the closest folder (this one or a parent) that contains a "data" folder
HERE = Path(__file__).resolve().parent
ROOT = next((p for p in [HERE, *HERE.parents] if (p / "data").is_dir()), HERE)
CSV_PATH = ROOT / "data" / "raw" / "unemployment_2015_2026.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SERIES_ID = "LNS14000000"          # BLS unemployment rate (seasonally adjusted)

TEST_SIZES = [12, 24, 36]          # test-window lengths in months (robustness check)
PRIMARY_TEST_SIZE = 24             # the window used for the saved CSV and plot
MIN_TRAIN = 36                     # skip a window if it leaves fewer training months


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_series(path: Path, series_id: str) -> pd.Series:
    """Load one BLS series as a regular monthly pandas Series (NaN = not published)."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["series_id"] == series_id].copy()
    if df.empty:
        raise ValueError(f"No rows found for series_id {series_id}")

    # BLS writes "-" for months with no published value (e.g. Oct 2025). -> NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    y = df.set_index("date")["value"].sort_index()
    y = y[~y.index.duplicated(keep="last")]
    y = y.asfreq("MS")                 # regular month-start index
    y.name = "unemployment_rate"
    return y


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def naive_forecast(train: pd.Series) -> float:
    """Next month = the most recent published value (skips months with no value)."""
    return float(train.dropna().iloc[-1])


def expanding_window_forecast(y: pd.Series, test_size: int) -> pd.DataFrame:
    """For each of the last `test_size` months, forecast it from the months before it."""
    dates, actual, pred = [], [], []
    for t in range(len(y) - test_size, len(y)):
        dates.append(y.index[t])
        actual.append(y.iloc[t])
        pred.append(naive_forecast(y.iloc[:t]))

    out = pd.DataFrame(
        {"actual": actual, "naive": pred},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    # Months BLS did not publish have no actual value to score against
    return out.dropna(subset=["actual"])


def score(res: pd.DataFrame):
    err = res["actual"] - res["naive"]
    return float(np.sqrt((err**2).mean())), float(err.abs().mean())


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def main():
    y = load_series(CSV_PATH, SERIES_ID)

    print(f"Series      : {SERIES_ID}")
    print(f"Range       : {y.index.min():%Y-%m} to {y.index.max():%Y-%m} ({len(y)} months)")
    missing = y[y.isna()].index
    if len(missing):
        print("Missing     : " + ", ".join(f"{d:%Y-%m}" for d in missing)
              + "  (left as missing, not scored)")

    rows, primary = [], None
    for ts in TEST_SIZES:
        if len(y) - ts < MIN_TRAIN:
            print(f"Skipping {ts}-month window: leaves under {MIN_TRAIN} training months")
            continue
        res = expanding_window_forecast(y, ts)
        rmse, mae = score(res)
        rows.append({"test_months": ts, "n_scored": len(res), "RMSE": rmse, "MAE": mae})
        if ts == PRIMARY_TEST_SIZE:
            primary = res

    table = pd.DataFrame(rows)
    print("\nNaive results (lower is better):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.to_csv(OUT_DIR / "naive_baseline_scores.csv", index=False)

    if primary is not None:
        primary.to_csv(OUT_DIR / "naive_baseline_forecasts.csv")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(primary.index, primary["actual"], marker="o", label="Actual")
        ax.plot(primary.index, primary["naive"], marker="s", linestyle="--", label="Naive")
        ax.set_title(f"US unemployment rate: one-step-ahead naive, last {PRIMARY_TEST_SIZE} months")
        ax.set_ylabel("Percent")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "naive_baseline_plot.png", dpi=150)
        print(f"\nSaved naive_baseline_scores.csv, naive_baseline_forecasts.csv, "
              f"naive_baseline_plot.png to {OUT_DIR}")


if __name__ == "__main__":
    main()