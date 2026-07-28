'''Module to pull and cache rates data'''
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)

RATE_FILE = DATA_DIR / 'rates.csv'

FRED_SERIES = {
    'DGS1MO': 30,
    'DGS3MO': 90,
    'DGS6MO': 180,
    'DGS1': 365,
}

TARGET_DTE = [
    30,
    60,
    90,
    120,
    180,
    270,
    365,
]

# Cache

def load_rates() -> pd.DataFrame:
    """
    Load cached Treasury data.

    Returns
    -------
    DataFrame
        Indexed by date.
    """

    if not RATE_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        RATE_FILE,
        parse_dates=['date']
    )

    return df


def pull_rates(start, end):
    """
    Download Treasury yields from FRED.
    """

    frames = []

    for series in FRED_SERIES:

        df = DataReader(
            series,
            'fred',
            start,
            end
        )

        frames.append(df)

    rates = pd.concat(frames, axis=1)

    rates = rates.ffill()

    rates = (
        rates
        .reset_index()
        .rename(columns={'DATE': 'date'})
    )

    return rates

def update_cache(start, end):
    """
    Ensure the cache contains the requested date range.
    """

    cached = load_rates()

    if cached.empty:

        rates = pull_rates(start, end)

        rates.to_csv(
            RATE_FILE,
            index=False
        )

        return rates

    cache_start = cached['date'].min()
    cache_end = cached['date'].max()

    pieces = [cached]

    if start < cache_start:

        new = pull_rates(
            start,
            cache_start - pd.Timedelta(days=1)
        )

        pieces.append(new)

    if end > cache_end:

        new = pull_rates(
            cache_end + pd.Timedelta(days=1),
            end
        )

        pieces.append(new)

    rates = (
        pd.concat(pieces)
        .drop_duplicates('date')
        .sort_values('date')
    )

    rates.to_csv(
        RATE_FILE,
        index=False
    )

    return rates

# Interpolate if necessary

def interpolate_rate(row):
    """
    Returns annualized Treasury yield for target DTE.

    Output is decimal (0.042 instead of 4.2).
    """

    dte = row['target_dte']

    if dte == 30:
        return row['DGS1MO'] / 100

    if dte == 90:
        return row['DGS3MO'] / 100

    if dte == 180:
        return row['DGS6MO'] / 100

    if dte == 365:
        return row['DGS1'] / 100

    if dte == 60:

        w = (60 - 30) / (90 - 30)

        return (
            row['DGS1MO']
            + w * (row['DGS3MO'] - row['DGS1MO'])
        ) / 100

    if dte == 120:

        w = (120 - 90) / (180 - 90)

        return (
            row['DGS3MO']
            + w * (row['DGS6MO'] - row['DGS3MO'])
        ) / 100

    if dte == 270:

        w = (270 - 180) / (365 - 180)

        return (
            row['DGS6MO']
            + w * (row['DGS1'] - row['DGS6MO'])
        ) / 100

    raise ValueError(f'Unsupported DTE {dte}')


def merge_rates(df: pd.DataFrame):
    """
    Adds a risk-free rate column based on date
    and target_dte.

    Returns
    -------
    DataFrame
    """

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    start = df['date'].min()
    end = df['date'].max()

    rates = update_cache(start, end)

    # Normalize dates
    df['date'] = pd.to_datetime(df['date'])
    rates['date'] = pd.to_datetime(rates['date'])

    merged = df.merge(
        rates,
        on='date',
        how='left'
    )

    merged['r'] = merged.apply(
        interpolate_rate,
        axis=1
    )

    merged = merged.drop(
        columns=list(FRED_SERIES.keys())
    )

    return merged