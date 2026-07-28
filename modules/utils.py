import pandas as pd
import numpy as np

def select_otm(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (
            (df['target_moneyness'] < 1)
            & (df['option_type'] == 'put')
        )
        |
        (
            (df['target_moneyness'] >= 1)
            & (df['option_type'] == 'call')
        )
    ]

def create_surface(df: pd.DataFrame, date: str) -> pd.DataFrame:
    df['log_moneyness'] = np.log(df['target_moneyness'])
    surface = (
        df[df['date'] == date]
        .pivot_table(
            index='target_dte',
            columns='log_moneyness',
            values='IV',
            aggfunc='mean'
        )
    )

    return surface

def interpolate_surface(surface: pd.DataFrame) -> pd.DataFrame:

    return (
        surface
        .interpolate(axis=1)
        .interpolate(axis=0)
    )