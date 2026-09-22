import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_gap(S, K1, K2, T, r, sigma, option_type='call'):
    """
    Valuación de una opción GAP Call o Put utilizando el modelo de Black-Scholes adaptado.

    Parámetros:
    S : float
        Precio actual del subyacente.
    K1 : float
        Primer precio de ejercicio (strike).
    K2 : float
        Segundo precio de ejercicio (precio al que se compara S en el vencimiento).
    T : float
        Tiempo hasta el vencimiento (en años).
    r : float
        Tasa de interés libre de riesgo anual.
    sigma : float
        Volatilidad anual del subyacente.
    option_type : str
        'call' para una opción de compra, 'put' para una opción de venta.
    """
    d1 = (np.log(S / K2) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) * np.exp((r - r) * T) - K1 * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K1 * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) * np.exp((r - r) * T)
    else:
        raise ValueError("option_type debe ser 'call' o 'put'")

    return price

# Parámetros proporcionados
subyacente = 17.00  # Precio actual del subyacente
strike1 = 17.00     # Primer strike (K1)
strike2 = 16.00     # Segundo strike (K2)
volatilidad = 0.04  # Volatilidad anual (4%)
vencimiento = 6/12  # Vencimiento en años (6 meses)
tasa_libre_riesgo = 0.04 # Tasa de interés libre de riesgo anual (4%)

# Valuar una opción GAP Call
valor_opcion_call_gap = black_scholes_gap(subyacente, strike1, strike2, vencimiento, tasa_libre_riesgo, volatilidad, option_type='call')
print(f"Valor de la opción GAP Call: {valor_opcion_call_gap:.4f}")

# Valuar una opción GAP Put (ejemplo, aunque el problema solo pide valuar la opción)
valor_opcion_put_gap = black_scholes_gap(subyacente, strike1, strike2, vencimiento, tasa_libre_riesgo, volatilidad, option_type='put')
print(f"Valor de la opción GAP Put: {valor_opcion_put_gap:.4f}")

# Definir un rango de precios del subyacente al vencimiento (S_T)
# Se ajusta el rango alrededor del strike1 para una mejor visualización
S_T_range = np.linspace(strike1 - 10, strike1 + 10, 100)

# Payout fijo para opciones binarias (ejemplo: 1 unidad monetaria)
payout = 1

# Calcular los pagos para una opción binaria 'cash-or-nothing' call
payoff_binary_call = np.where(S_T_range > strike1, payout, 0)

# Calcular los pagos para una opción binaria 'cash-or-nothing' put
payoff_binary_put = np.where(S_T_range < strike1, payout, 0)

# Visualización
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(S_T_range, payoff_binary_call, label='Payoff Opción Binaria Call')
plt.axvline(strike1, color='r', linestyle='--', label=f'Strike K1 = {strike1}')
plt.title('Patrón de Pago de Opción Binaria Call')
plt.xlabel('Precio del Subyacente al Vencimiento (S_T)')
plt.ylabel('Pago')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(S_T_range, payoff_binary_put, label='Payoff Opción Binaria Put', color='orange')
plt.axvline(strike1, color='r', linestyle='--', label=f'Strike K1 = {strike1}')
plt.title('Patrón de Pago de Opción Binaria Put')
plt.xlabel('Precio del Subyacente al Vencimiento (S_T)')
plt.ylabel('Pago')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
