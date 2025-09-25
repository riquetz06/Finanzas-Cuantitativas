import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

# -------------------------------
# Parámetros del problema
# -------------------------------
S0 = 24.0         # Precio spot
K_call = 22.0     # Strike Call
K_put = 25.0      # Strike Put
q = 0.01          # Dividendos (rendimiento continuo)
r = 0.05          # Tasa libre de riesgo
sigma = 0.25      # Volatilidad
T_call = 8/12     # Tiempo hasta vencimiento Call (años)
T_put = 4/12      # Tiempo hasta vencimiento Put (años)
t1 = 2/12         # Tiempo de elección (años)

# -------------------------------
# Fórmulas de Black-Scholes
# -------------------------------
def bs_call_price(S, K, tau, r, q, sigma):
    """Precio de una opción Call europea (Black-Scholes)"""
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bs_put_price(S, K, tau, r, q, sigma):
    """Precio de una opción Put europea (Black-Scholes)"""
    if tau <= 0:
        return np.maximum(K - S, 0.0)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)

# -------------------------------
# Simulación Monte Carlo
# -------------------------------
np.random.seed(42)          # Semilla para reproducibilidad
n_sims = 200_000            # Número de simulaciones

# Simular el precio del subyacente en t1
z = np.random.normal(size=n_sims)
S_t1 = S0 * np.exp((r - q - 0.5 * sigma**2) * t1 + sigma * np.sqrt(t1) * z)

# Precios de call y put en t1
calls_t1 = bs_call_price(S_t1, K_call, T_call - t1, r, q, sigma)
puts_t1  = bs_put_price(S_t1, K_put, T_put - t1, r, q, sigma)


# Valor de la chooser en t1 (elige la más valiosa)
chooser_t1 = np.maximum(calls_t1, puts_t1)

# Precio hoy = valor esperado descontado
chooser_price_complex = np.exp(-r * t1) * np.mean(chooser_t1)
std_error_complex = np.exp(-r * t1) * np.std(chooser_t1) / np.sqrt(n_sims)

# -------------------------------
# Resultados
# -------------------------------
print(f"Precio de la opción chooser compleja: {chooser_price_complex:.6f}")
print(f"Error estándar Monte Carlo: {std_error_complex:.6f}")
