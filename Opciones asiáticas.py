precio_inicial = 20.00
tiempo_hasta_vencimiento = 2.0
tasa_interes_libre_riesgo = 0.045
tasa_dividendos = 0.01
volatilidad = 0.05
vida_esperada = 4.0
volatilidad_compania = 0.20

import numpy as np
import random

num_simulations = 10000
num_steps = 252 * int(tiempo_hasta_vencimiento)  # Assuming 252 trading days per year
dt = tiempo_hasta_vencimiento / num_steps

price_paths = np.zeros((num_simulations, num_steps + 1))
price_paths[:, 0] = precio_inicial

for step in range(1, num_steps + 1):
    z = np.random.standard_normal(num_simulations)
    price_paths[:, step] = price_paths[:, step - 1] * np.exp((tasa_interes_libre_riesgo - tasa_dividendos - 0.5 * volatilidad**2) * dt + volatilidad * np.sqrt(dt) * z)

average_prices = np.mean(price_paths, axis=1)
strike_price = precio_inicial
payoffs = np.maximum(0, average_prices - strike_price)

discount_factor = np.exp(-tasa_interes_libre_riesgo * tiempo_hasta_vencimiento)
discounted_payoffs = payoffs * discount_factor
asian_option_value = np.mean(discounted_payoffs)

print(f"Estimated Asian Option Value: {asian_option_value:.4f}")
print(f"El valor calculado de la opción asiática es: {asian_option_value}")
