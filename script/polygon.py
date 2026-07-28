import os
import json
import time
from typing import Any, Dict, List, Optional, Union
import yfinance as yf
import requests
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime, timedelta


# ==========================
# API SETUP
# ==========================
load_dotenv()
API_KEY = os.getenv('POLYGON_KEY')

if API_KEY is None:
    raise ValueError('Missing POLYGON_KEY')

BASE = 'https://api.polygon.io'
BASE_URL = 'https://api.massive.com/v1/open-close'

# ==========================
# CONFIG
# ==========================
UNDERLYING = 'SPY'

# ~Half a year back from today. The script produced more days that this
# because it had to be ran over multiple days since it took so long
# thus pushing the end date farther back by 3 days.
# END_DATE = datetime.today().date()
# START_DATE = END_DATE - timedelta(days=180)
# START_DATE = START_DATE.strftime('%Y-%m-%d')
# END_DATE = END_DATE.strftime('%Y-%m-%d')
# Hard code start and end dates when re-running script
END_DATE = '2026-07-15'
START_DATE = '2026-01-15'

# Surface grid for efficiency of API calls
TARGET_MONEYNESS = [
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15
]

TARGET_DTE = [
    30,
    60,
    90,
    120,
    180,
    270,
    365
]

MAX_DTE_ERROR = 20
MIN_VOLUME = 25

# Saving
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# v2 run
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'spy_surface_checkpoint_v2.csv')
COMPLETED_DATES_FILE = os.path.join(OUTPUT_DIR, 'completed_dates_v2.txt')
FINAL_FILE = os.path.join(OUTPUT_DIR, 'spy_surface_raw_v2.csv')

CHAIN_CACHE = os.path.join(OUTPUT_DIR, 'chain_cache')
os.makedirs(CHAIN_CACHE, exist_ok=True)
SPY_PRICE_FILE = os.path.join(
    OUTPUT_DIR,
    'spy_prices.csv'
)

# ==========================
# API HELPERS
# ==========================
def request_with_retry(
        url: str,
        params: Dict[str, Any],
        max_retry: int = 10
) -> Dict[str, Any]:
    """
    Perform an HTTP GET request and retry on transient Polygon API rate limits.

    Args:
    url: The full request URL.
    params: Query parameters for the request.
    max_retry: Maximum number of retry attempts for rate-limited responses.

    Returns:
    The decoded JSON response body.

    Raises:
    Exception: If the maximum retry count is exceeded.
    """

    retry = 0
    while retry < max_retry:
        r = requests.get(url, params=params)

        # If API rate limit is hit, cool off for a little bit longer
        if r.status_code == 429:
            wait = 15
            print(f'Rate limit. Sleeping {wait}s')
            time.sleep(wait)
            retry += 1
            continue
        else:
            # Sleep for 12 seconds after each call to adhere to 5 calls/min API limit
            time.sleep(12)

        r.raise_for_status()
        return r.json()
    raise Exception('Maximum retries exceeded')

def get_option_chain(date: str) -> Dict[str, Any]:
    """
    Fetch or load the cached option contract chain for a given date.

    Args:
    date: The target trade date in YYYY-MM-DD format.

    Returns:
    A dictionary containing the option chain results.
    """
    cache_file = os.path.join(CHAIN_CACHE, f'{date}.json')

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    url = f'{BASE}/v3/reference/options/contracts'
    params = {
        'underlying_ticker': UNDERLYING,
        'as_of': date,
        'expiration_date.gte': (
            pd.Timestamp(date) +
            pd.Timedelta(days=14)
        ).strftime('%Y-%m-%d'),
        'expiration_date.lte': (
            pd.Timestamp(date)
            +
            pd.Timedelta(days=400)
        ).strftime('%Y-%m-%d'),
        'limit': 1000,
        'apiKey': API_KEY
    }

    results = []

    # Parse through pages of json data until the last page is reached
    while url:
        data = request_with_retry(url, params)

        results.extend(data.get('results', []))
        url = data.get('next_url')
        params = {'apiKey': API_KEY}

    output = {'results': results}

    with open(cache_file, 'w') as f:
        json.dump(output, f)

    return output
    
def get_option_ohlc(ticker: str, date: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve OHLC data for a specific option contract on a date.

    Args:
    ticker: The option contract ticker symbol.
    date: The trade date in YYYY-MM-DD format.

    Returns:
    A dictionary with OHLC and volume information, or None if unavailable.
    """

    url = f'{BASE_URL}/{ticker}/{date}'
    params = {'adjusted': 'true', 'apiKey': API_KEY}

    try:
        data = request_with_retry(url, params)
    except Exception:
        return None

    if data.get('status') != 'OK':
        return None
    return data

def get_spy_history(start: str, end: str) -> Dict[pd.Timestamp, float]:
    """
    Download historical SPY close prices over a date range.

    Args:
    start: Start date in YYYY-MM-DD format.
    end: End date in YYYY-MM-DD format.

    Returns:
    A mapping from normalized pandas Timestamp dates to closing prices.
    """

    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()

    if os.path.exists(SPY_PRICE_FILE):
        spy_df = pd.read_csv(SPY_PRICE_FILE)
        spy_df['date'] = pd.to_datetime(
            spy_df['date'],
            utc=True
        ).dt.date

    else:
        spy_df = pd.DataFrame(columns=['date', 'c'])

    # Download only if requested end date is missing
    if (spy_df.empty or end not in set(spy_df['date'])):
        if spy_df.empty: 
            download_start = start
        else:
            last_date = spy_df['date'].max()
            download_start = (last_date + pd.Timedelta(days=1))

        print(f'Downloading SPY prices from {download_start} to {end}')

        spy = yf.Ticker(UNDERLYING)

        new = spy.history(
            start=download_start.strftime('%Y-%m-%d'),
            end=(end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            auto_adjust=True
        )

        if not new.empty:
            new = (new.reset_index()[['Date', 'Close']]
                .rename(
                    columns={
                        'Date': 'date',
                        'Close': 'c'
                    }
                )
            )

            new['date'] = pd.to_datetime(
                new['date'],
                utc=True
            ).dt.date

            spy_df = (
                pd.concat(
                    [spy_df, new],
                    ignore_index=True
                )
                .drop_duplicates(
                    subset='date'
                )
                .sort_values('date')
            )

            spy_df.to_csv(
                SPY_PRICE_FILE,
                index=False
            )

    # Return only requested window
    spy_df = spy_df[
        (spy_df['date'] >= start)
        & (spy_df['date'] <= end)
    ]

    return dict(zip(
        pd.to_datetime(spy_df['date']),
        spy_df['c']
    ))

# ==========================
# SURFACE SELECTION
# ==========================
def select_surface_contracts(
        contracts: List[Dict[str, Any]],
        spot: float,
        date: Union[str, pd.Timestamp, datetime]
) -> List[Dict[str, Any]]:
    """
    Choose option contracts close to the target surface grid.

    Args:
    contracts: Raw contract metadata from the Polygon option chain.
    spot: The underlying SPY spot price.
    date: The trade date for DTE calculation.

    Returns:
    A filtered list of selected contract metadata dictionaries.
    """

    candidates = []
    date = pd.Timestamp(date)
    for c in contracts:

        strike = c.get('strike_price')
        expiry = c.get('expiration_date')
        ticker = c.get('ticker')
        option_type = c.get('contract_type')

        if (
            strike is None
            or expiry is None
            or ticker is None
            or option_type is None
        ):
            continue

        expiry = pd.Timestamp(expiry)

        dte = (expiry - date).days
        if dte <= 0: continue

        moneyness = strike / spot

        dte_distance = [(abs(dte - target), target) for target in TARGET_DTE]
        closest_dte = min(dte_distance)[1]

        # Limit options to those close a DTE on the grid
        if abs(dte - closest_dte) > MAX_DTE_ERROR: continue

        money_distance = [(abs(moneyness - target), target) for target in TARGET_MONEYNESS]
        closest_money = min(money_distance)[1]

        # Limit options those close to a moneyness on the grid
        if abs(moneyness - closest_money) > 0.03: continue

        candidates.append({
            'ticker': ticker,
            'strike': strike,
            'expiry': expiry.date(),
            'dte': dte,
            'years_to_expiry': dte / 365.25,
            'moneyness': moneyness,
            'target_moneyness': closest_money,
            'target_dte': closest_dte,
            'option_type': option_type
        })

    if not candidates:
        return []

    df = pd.DataFrame(candidates)

    # Distance from ideal surface grid point
    df['distance'] = (
        abs(df['moneyness'] - df['target_moneyness'])
        + abs(df['dte'] - df['target_dte']) / 365
    )

    # One contract per:
    # call/put x moneyness x maturity bucket
    df = (df.sort_values('distance').drop_duplicates(
        subset=[
            'option_type',
            'target_moneyness',
            'target_dte'
        ]))

    return df.drop(columns=['distance']).to_dict('records')

# ==========================
# CHECKPOINT
# ==========================
if os.path.exists(CHECKPOINT_FILE):
    checkpoint = pd.read_csv(CHECKPOINT_FILE)
    rows = checkpoint.to_dict('records')
    completed = set(zip(checkpoint['date'].astype(str), checkpoint['ticker']))
    print(f'Loaded {len(rows)} rows')

else:
    rows = []
    completed = set()

# Load completed dates
if os.path.exists(COMPLETED_DATES_FILE):
    with open(COMPLETED_DATES_FILE, 'r') as f:
        completed_dates = set(line.strip() for line in f.readlines())
else:
    completed_dates = set()

# ==========================
# MAIN LOOP
# ==========================
print('Loading SPY history')

spy_map = get_spy_history(START_DATE, END_DATE)
dates = pd.bdate_range(START_DATE, END_DATE)

pbar_outer = tqdm(
    dates,
    total=len(dates),
    initial=len(completed_dates),
    desc='Collecting surface',
    position=0
)

for date in pbar_outer:
    date_str = str(date.date())

    # Check if the date has been processed
    if date_str in completed_dates: continue

    spot = spy_map.get(date)

    if spot is None: continue

    chain = get_option_chain(str(date.date()))

    contracts = select_surface_contracts(chain.get('results', []), spot, date)

    print('\n')
    print(date.date(), 'chain:', len(chain['results']), 'selected:', len(contracts))

    day_start = len(rows)
    day_saved = 0

    pbar_inner = tqdm(
        contracts, 
        desc=f'Retrieving contracts for {date}', 
        leave=False,
        position=1
    )
    for contract in pbar_inner:
        ticker = contract['ticker']
        key = (str(date.date()), ticker)

        if key in completed: continue

        data = get_option_ohlc(ticker, str(date.date()))

        if data is None: continue

        volume = data.get('volume')
        close = data.get('close')

        if (
            volume is None
            or volume < MIN_VOLUME
            or close is None
            or close <= 0
        ):
            continue

        rows.append({
            'date': date.date(),
            'spot': spot,
            **contract, 
            'close': close,
            'volume': volume
        })

        day_saved += 1
        completed.add(key)
        pbar_inner.update(1)

    completed_dates.add(date_str)
    pbar_outer.update(1)
    with open(COMPLETED_DATES_FILE, 'w') as f:
        for d in sorted(completed_dates):
            f.write(d + '\n')
    
    print(date.date(), 'saved:', len(rows) - day_start, '/', len(contracts))
    print('\n')

    # Checkpoint after every day
    pd.DataFrame(rows).to_csv(CHECKPOINT_FILE, index=False)

pbar_inner.close()
pbar_outer.close()

# ==========================
# FINAL SAVE
# ==========================
df = pd.DataFrame(rows)
df.to_csv(FINAL_FILE, index=False)

print(f'Saved {len(df)} rows')
print(FINAL_FILE)