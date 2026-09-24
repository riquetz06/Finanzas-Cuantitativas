import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def d_param(S, K, r, q, sigma, T):
    """Calcula el parámetro d del modelo Reiner y Rubinstein (1991)."""
    numerator = np.log(S / K) + (r - q - (sigma**2) / 2) * T
    denominator = sigma * np.sqrt(T)
    return numerator / denominator

def cash_or_nothing_call(S, K, Q, r, q, sigma, T):
    """Valora una opción Call Cash-or-Nothing."""
    d = d_param(S, K, r, q, sigma, T)
    return Q * np.exp(-r * T) * norm.cdf(d)

def cash_or_nothing_put(S, K, Q, r, q, sigma, T):
    """Valora una opción Put Cash-or-Nothing."""
    d = d_param(S, K, r, q, sigma, T)
    return Q * np.exp(-r * T) * norm.cdf(-d)

# ==========================================
# Parámetros del Modelo
# ==========================================
K = 100.0     # Precio de ejercicio (Strike)
Q = 50.0      # Pago fijo en efectivo (Cash Payoff)
r = 0.05      # Tasa libre de riesgo (5%)
q = 0.02      # Tasa de dividendos (2%)
sigma = 0.20  # Volatilidad (20%)
T = 1.0       # Tiempo al vencimiento (1 año)

# Rango de precios del subyacente (S) para graficar
S_range = np.linspace(50, 150, 400)

# ==========================================
# Cálculos de Precios y Payoffs
# ==========================================
# Precios actuales según la fórmula
call_prices = cash_or_nothing_call(S_range, K, Q, r, q, sigma, T)
put_prices = cash_or_nothing_put(S_range, K, Q, r, q, sigma, T)

# Perfil de pago al vencimiento (Payoff)
payoff_call = np.where(S_range > K, Q, 0)
payoff_put = np.where(S_range < K, Q, 0)

# ==========================================
# Gráficas
# ==========================================
plt.figure(figsize=(12, 5))

# Grafico 1: Call Cash-or-Nothing
plt.subplot(1, 2, 1)
plt.plot(S_range, payoff_call, 'r--', label='Payoff al vencimiento (T=0)', linewidth=1.5)
plt.plot(S_range, call_prices, 'b-', label=f'Valor actual (T={T} años)', linewidth=2)
plt.axvline(x=K, color='gray', linestyle=':', label=f'Strike K = {K}')
plt.title('Opción Call Cash-or-Nothing')
plt.xlabel('Precio del Subyacente (S)')
plt.ylabel('Valor de la Opción / Payoff')
plt.grid(True, alpha=0.3)
plt.legend()

# Grafico 2: Put Cash-or-Nothing
plt.subplot(1, 2, 2)
plt.plot(S_range, payoff_put, 'r--', label='Payoff al vencimiento (T=0)', linewidth=1.5)
plt.plot(S_range, put_prices, 'g-', label=f'Valor actual (T={T} años)', linewidth=2)
plt.axvline(x=K, color='gray', linestyle=':', label=f'Strike K = {K}')
plt.title('Opción Put Cash-or-Nothing')
plt.xlabel('Precio del Subyacente (S)')
plt.ylabel('Valor de la Opción / Payoff')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
