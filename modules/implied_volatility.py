'''Given inputs, calculate implied volatility of an option'''
from modules.black_scholes import black_scholes

def bisection(
    market_price: float,
    S: float, 
    T: float, 
    r: float, 
    K: float, 
    option_type='call',
    tol: float=1e-6,
    max_iter: int=100,
):
    # Declare intial high,low guesses
    low, high = 1e-6, 15

    for _ in range(max_iter):
        sigma = (low + high)/2
        price = black_scholes(
            S=S,
            T=T,
            r=r,
            K=K,
            sigma=sigma,
            option_type=option_type
        )

        if abs(price - market_price) < tol:
            return sigma

        if price > market_price:
            high = sigma

        else:
            low = sigma

    raise RuntimeError('Bisection method failed to converge.')
