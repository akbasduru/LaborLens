import json

import pandas as pd  # type: ignore
import requests  # type: ignore

headers = {'Content-type': 'application/json'}
data = json.dumps({
    "seriesid": ["CES0000000001", "LNS14000000", "JTU110099000000000HIL"],   # Nonfarm payroll employment, Unemployment rate and Job Openings and Labor Turnover Survey
    "startyear": "2020",
    "endyear": "2026",
    "registrationkey": "a9263bee36ce47d5a37eb6eaac684c27"
})

response = requests.post(
    'https://api.bls.gov/publicAPI/v2/timeseries/data/',
    data=data,
    headers=headers
)

results = response.json()

# Build a list of clean rows from the nested JSON
rows = []
for series in results['Results']['series']:
    series_id = series['seriesID']
    for entry in series['data']:
        rows.append({
            'series_id': series_id,
            'year': entry['year'],
            'period': entry['period'],
            'period_name': entry['periodName'],
            'value': entry['value']
        })

# Convert to a DataFrame
df = pd.DataFrame(rows)

# Create a proper date column (helps a LOT for time-series work later)
# BLS periods are like 'M01' through 'M12' for months
df['month'] = df['period'].str.replace('M', '').astype(int)
df['date'] = pd.to_datetime(df['year'] + '-' + df['month'].astype(str) + '-01')

# Sort chronologically, oldest first (BLS returns newest first by default)
df = df.sort_values(['series_id', 'date']).reset_index(drop=True)

# Reorder columns nicely
df = df[['series_id', 'date', 'year', 'period', 'period_name', 'value']]

# Save to CSV
df.to_csv('bls_data.csv', index=False)
print("Saved to bls_data.csv")
print(df.head())