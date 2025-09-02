import numpy as np
import scipy.stats as si

# Función Black-Scholes
def black_scholes(S, K, T, r, sigma, option="call"):
    """
    S : precio spot del subyacente
    K : precio de ejercicio (strike)
    T : tiempo hasta vencimiento en años
    r : tasa libre de riesgo
    sigma : volatilidad del subyacente
    option : "call" o "put"
    """
    # d1 y d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option == "call":
        precio = (S * si.norm.cdf(d1, 0.0, 1.0) - 
                  K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option == "put":
        precio = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - 
                  S * si.norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("La opción debe ser 'call' o 'put'")
    
    return precio


# Ejemplo de uso
S = 100     # precio spot
K = 105     # strike
T = 1       # tiempo a vencimiento (1 año)
r = 0.05    # tasa libre de riesgo (5%)
sigma = 0.2 # volatilidad (20%)

precio_call = black_scholes(S, K, T, r, sigma, option="call")
precio_put  = black_scholes(S, K, T, r, sigma, option="put")

print(f"Precio Call: {precio_call:.4f}")
print(f"Precio Put : {precio_put:.4f}")

