import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2026, 1, 1)


def get_fred_series(series_id: str, start_date: datetime.datetime,
                    end_date: datetime.datetime) -> pd.DataFrame:
    """Download one FRED series and limit it to the requested date range."""
    response = requests.get(
        'https://fred.stlouisfed.org/graph/fredgraph.csv',
        params={'id': series_id},
        timeout=30,
    )
    response.raise_for_status()

    data = pd.read_csv(
        StringIO(response.text),
        parse_dates=['observation_date'],
        index_col='observation_date',
    )
    return data.loc[start_date:end_date]


fed_funds = get_fred_series('FEDFUNDS', start, end) # Get the Federal Funds Rate series from FRED, monthly data, percentage
cpi = get_fred_series('CPIAUCSL', start, end) # Get the Consumer Price Index series from FRED, monthly data, index
claims = get_fred_series('ICSA', start, end) # Get the Initial Claims series from FRED, weekly data, number
sentiment = get_fred_series('UMCSENT', start, end) # Get the University of Michigan Sentiment series from FRED, monthly data, index


data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

fed_funds.to_csv(data_folder / "fed_funds.csv")
cpi.to_csv(data_folder / "cpi.csv")
claims.to_csv(data_folder / "claims.csv")
sentiment.to_csv(data_folder / "sentiment.csv")

print("All data saved successfully!")