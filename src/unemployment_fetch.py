import json
from pathlib import Path

import pandas as pd  # type: ignore
import requests  # type: ignore

API_KEY = "a9263bee36ce47d5a37eb6eaac684c27"

headers = {'Content-type': 'application/json'}
data = json.dumps({
    "seriesid": ["LNS14000000"],      # Unemployment rate, seasonally adjusted
    "startyear": "2015",
    "endyear": "2026",
    "registrationkey": API_KEY
})

response = requests.post(
    'https://api.bls.gov/publicAPI/v2/timeseries/data/',
    data=data,
    headers=headers,
    timeout=60
)
results = response.json()

if results.get('status') != 'REQUEST_SUCCEEDED':
    raise RuntimeError(f"BLS request failed: {results.get('message')}")

rows = []
for series in results['Results']['series']:
    series_id = series['seriesID']
    for entry in series['data']:
        if entry['period'] == 'M13':      # annual average, not a month
            continue
        rows.append({
            'series_id': series_id,
            'year': entry['year'],
            'period': entry['period'],
            'period_name': entry['periodName'],
            'value': entry['value']
        })

df = pd.DataFrame(rows)
df['month'] = df['period'].str.replace('M', '').astype(int)
df['date'] = pd.to_datetime(df['year'] + '-' + df['month'].astype(str) + '-01')
df = df.sort_values(['series_id', 'date']).reset_index(drop=True)
df = df[['series_id', 'date', 'year', 'period', 'period_name', 'value']]

# Save where naive_model.py and arima_model.py look for it
out_path = Path(__file__).resolve().parent
if out_path.name == "src":
    out_path = out_path.parent
out_path = out_path / "data" / "raw" / "unemployment_2015_2026.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(out_path, index=False)
print(f"Saved {len(df)} months to {out_path}")
print(df.head())