'''Given inputs, calculate implied volatility of an option'''
from modules.black_scholes import black_scholes
import numpy as np

def bisection(
    market_price: float,
    S: float, 
    T: float, 
    r: float, 
    K: float, 
    option_type='call',
    tol: float=1e-6,
    max_iter: int=30,
):
    # Declare intial high,low guesses
    low = np.full(len(S), 1e-6, dtype=float)
    high = np.full(len(S), 5.0, dtype=float)

    for _ in range(max_iter):
        sigma = (low + high)/2
        prices = black_scholes(
            S=S,
            T=T,
            r=r,
            K=K,
            sigma=sigma,
            option_type=option_type
        )

        error = np.abs(prices - market_price)

        mask = prices > market_price

        high[mask] = sigma[mask]
        low[~mask] = sigma[~mask]

    return sigma
