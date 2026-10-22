"""
Standard ARIMA baseline for the US unemployment rate (BLS series LNS14000000).

Input : bls_data.csv, produced by your BLS fetch script
        (columns: series_id, date, year, period, period_name, value)
Method: expanding-window, one-step-ahead ARIMA forecasts. At every test month the
        model is refit on all earlier months only, then predicts that month.
Output: RMSE / MAE for several test-window lengths, plus a CSV and a plot of the
        primary window.

Run:  python arima_baseline.py
Needs: pip install pandas numpy statsmodels matplotlib
"""

import warnings

import matplotlib

matplotlib.use("Agg")  # save plots to file, no window needed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = next((p for p in [HERE, *HERE.parents] if (p / "data").is_dir()), HERE)
CSV_PATH = ROOT / "data" / "raw" / "unemployment_2015_2026.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SERIES_ID = "LNS14000000"          # BLS unemployment rate (seasonally adjusted)

ORDER = (1, 1, 1)                  # (p, d, q) - the standard baseline
USE_AIC_SELECTION = False          # True = pick p and q by AIC (d stays as in ORDER)
MAX_P, MAX_Q = 3, 3                # search range if USE_AIC_SELECTION is True

TEST_SIZES = [12, 24, 36]          # test-window lengths in months (robustness check)
PRIMARY_TEST_SIZE = 24             # the window used for the saved CSV and plot
MIN_TRAIN = 36                     # skip a window if it leaves fewer training months


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_series(path: str, series_id: str) -> pd.Series:
    """Load one BLS series as a regular monthly pandas Series (NaN = not published)."""
    df = pd.read_csv(path)
    if "series_id" not in df.columns:  # file saved without a header row
        df = pd.read_csv(
            path,
            header=None,
            names=["series_id", "date", "year", "period", "period_name", "value"],
        )
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
def fit_arima(train: pd.Series, order):
    """Fit ARIMA. Missing months (NaN) are handled by the model, not filled in."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(train, order=order).fit()


def select_order(train: pd.Series, d: int, max_p: int, max_q: int):
    """Pick (p, d, q) with the lowest AIC on the training data only."""
    best_aic, best_order = np.inf, None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = fit_arima(train, (p, d, q)).aic
            except Exception:
                continue
            if aic < best_aic:
                best_aic, best_order = aic, (p, d, q)
    if best_order is None:
        raise RuntimeError("No ARIMA order could be fitted")
    return best_order


def expanding_window_forecast(y: pd.Series, test_size: int, order) -> pd.DataFrame:
    """Refit on all data before month t, forecast month t, for the last `test_size` months."""
    dates, actual, pred = [], [], []
    for t in range(len(y) - test_size, len(y)):
        fit = fit_arima(y.iloc[:t], order)
        dates.append(y.index[t])
        actual.append(y.iloc[t])
        pred.append(fit.forecast(steps=1).iloc[0])

    out = pd.DataFrame(
        {"actual": actual, "arima": pred},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    # Months BLS did not publish have no actual value to score against
    return out.dropna(subset=["actual"])


def score(res: pd.DataFrame):
    err = res["actual"] - res["arima"]
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
              + "  (left as missing, not filled)")

    # Stationarity check - helps decide d. Small p-value = stationary.
    clean = y.dropna()
    print(f"\nADF p-value, level      : {adfuller(clean)[1]:.3f}")
    print(f"ADF p-value, 1st diff   : {adfuller(clean.diff().dropna())[1]:.3f}")
    print("(p < 0.05 suggests stationary; if the level fails but the diff passes, try d=1)")

    # Choose the order using only months before the longest test window
    order = ORDER
    if USE_AIC_SELECTION:
        pre_test = y.iloc[: len(y) - max(TEST_SIZES)]
        order = select_order(pre_test, ORDER[1], MAX_P, MAX_Q)
    print(f"\nARIMA order : {order}")

    # Test several window lengths: a single holdout can be misleading
    rows, primary = [], None
    for ts in TEST_SIZES:
        if len(y) - ts < MIN_TRAIN:
            print(f"Skipping {ts}-month window: leaves under {MIN_TRAIN} training months")
            continue
        res = expanding_window_forecast(y, ts, order)
        rmse, mae = score(res)
        rows.append({"test_months": ts, "n_scored": len(res), "RMSE": rmse, "MAE": mae})
        if ts == PRIMARY_TEST_SIZE:
            primary = res

    table = pd.DataFrame(rows)
    print("\nResults (lower is better):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.to_csv(OUT_DIR / "arima_baseline_scores.csv", index=False)

    if primary is not None:
        primary.to_csv(OUT_DIR / "arima_baseline_forecasts.csv")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(primary.index, primary["actual"], marker="o", label="Actual")
        ax.plot(primary.index, primary["arima"], marker="x", label=f"ARIMA{order}")
        ax.set_title(f"US unemployment rate: one-step-ahead ARIMA, last {PRIMARY_TEST_SIZE} months")
        ax.set_ylabel("Percent")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "arima_baseline_plot.png", dpi=150)
        print(f"\nSaved arima_baseline_scores.csv, arima_baseline_forecasts.csv, "
                f"arima_baseline_plot.png to {OUT_DIR}")

if __name__ == "__main__":
    main()