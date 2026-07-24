'''Given inputs, calculate implied volatility of an option'''
from modules.black_scholes import black_scholes

def bisection(
    S: float, 
    T: float, 
    r: float, 
    K: float, 
    option_type='call',
    tol: float=1e-6,
    max_iter: int=100,
):
    # Declare intial high,low guesses
    low, high = 1e-6, 5

    # Check to ensure price is feasible to solve
    price_low = black_scholes(S, T, r, K, low, option_type)
    price_high = black_scholes(S, T, r, K, high, option_type)

    if not (price_low <= S <= price_high):
        raise ValueError('Market price outside feasible range.')

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

        if abs(price - S) < tol:
            return sigma

        if price > S:
            high = sigma

        else:
            low = sigma

    raise RuntimeError('Bisection method failed to converge.')
