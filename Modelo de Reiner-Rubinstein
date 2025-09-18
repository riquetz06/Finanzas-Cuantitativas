import numpy as np
from scipy.stats import norm

def reiner_rubinstein_cash_or_nothing(S0, K, H, r, sigma, T, Q=1, option_type="call", barrier_type="down-out"):
    """
    Valuación de opciones exóticas tipo cash-or-nothing con barrera (Reiner & Rubinstein, 1991)
    
    Parámetros:
    S0 : float
        Precio spot del subyacente
    K : float
        Strike
    H : float
        Barrera
    r : float
        Tasa libre de riesgo
    sigma : float
        Volatilidad
    T : float
        Tiempo a vencimiento (en años)
    Q : float
        Pago fijo si la opción se activa
    option_type : str
        "call" o "put"
    barrier_type : str
        "down-in", "down-out", "up-in", "up-out"
    """
    # Parámetros intermedios
    lambda_ = (r + 0.5 * sigma**2) / sigma**2
    x1 = (np.log(S0 / K) / (sigma * np.sqrt(T))) + lambda_ * sigma * np.sqrt(T)
    x2 = (np.log(S0 / H) / (sigma * np.sqrt(T))) + lambda_ * sigma * np.sqrt(T)
    y1 = (np.log((H**2) / (S0 * K)) / (sigma * np.sqrt(T))) + lambda_ * sigma * np.sqrt(T)
    y2 = (np.log(H / S0) / (sigma * np.sqrt(T))) + lambda_ * sigma * np.sqrt(T)

    # Normalización para puts (simetría put-call)
    phi = 1 if option_type == "call" else -1

    # Casos
    if barrier_type == "down-out" and option_type == "call":
        price = Q * np.exp(-r*T) * (norm.cdf(x2) - (H/S0)**(2*lambda_-2) * norm.cdf(y2))
    elif barrier_type == "down-in" and option_type == "call":
        price = Q * np.exp(-r*T) * (norm.cdf(x1) - norm.cdf(x2) + (H/S0)**(2*lambda_-2) * (norm.cdf(y2) - norm.cdf(y1)))
    else:
        raise NotImplementedError("Este caso aún no está implementado. Extiende las fórmulas para up-in/up-out y puts.")
    
    return price

# Ejemplo numérico
S0 = 100     # Spot
K = 100      # Strike
H = 90       # Barrera
r = 0.05     # Tasa libre
sigma = 0.2  # Volatilidad
T = 1.0      # 1 año
Q = 10       # Paga 10 si se cumple

precio = reiner_rubinstein_cash_or_nothing(S0, K, H, r, sigma, T, Q, option_type="call", barrier_type="down-out")
print("Prima opción down-and-out cash-or-nothing call:", round(precio, 4))
