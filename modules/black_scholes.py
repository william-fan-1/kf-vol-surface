'''Given inputs, calculate the price of an option using Black-Scholes'''
import numpy as np
from scipy.stats import norm

def black_scholes(
    S: float, 
    T: float, 
    r: float, 
    K: float, 
    sigma: float, 
    option_type: str='call'
):
    d1 = _d1(
        S=S,
        T=T,
        r=r,
        K=K,
        sigma=sigma
    )
    d2 = _d2(
        d1=d1,
        T=T,
        sigma=sigma
    )
    
    call_price = (
        S * norm.cdf(d1)
        - K * np.exp(-r*T) * norm.cdf(d2)
    )

    put_price = (
        K * np.exp(-r*T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )

    return np.where(option_type == 'call', call_price, put_price)

def _d1(
    S: float, 
    T: float, 
    r: float, 
    K: float, 
    sigma: float
):
    return (
        np.log(S/K)
        + (r + 0.5*sigma**2)*T
    ) / (sigma*np.sqrt(T))

def _d2(
    d1: float, 
    T: float, 
    sigma: float
):
    return d1 - sigma*np.sqrt(T)