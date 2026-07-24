'''Given inputs, calculate the price of an option using Black-Scholes'''
import numpy as 
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
    if option_type == 'call':
        return (S*norm.cdf(d1) 
        - np.exp(-r*T)*K*norm.cdf(d2))
    elif option_type == 'put':
        return (np.exp(-r*T)*K*norm.cdf(-d2)
        - S*norm.cdf(-d1))

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