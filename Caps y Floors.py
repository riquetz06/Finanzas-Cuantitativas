import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# 1. CAPLET - BLACK 76
# ============================================================

def caplet_black76(N, delta, P, F, K, sigma, T):

    d1 = (
        np.log(F / K) + 0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    caplet = (
        N
        * delta
        * P
        * (
            F * norm.cdf(d1)
            - K * norm.cdf(d2)
        )
    )

    return caplet


# ============================================================
# 2. FLOORLET - BLACK 76
# ============================================================

def floorlet_black76(N, delta, P, F, K, sigma, T):

    d1 = (
        np.log(F / K) + 0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    floorlet = (
        N
        * delta
        * P
        * (
            K * norm.cdf(-d2)
            - F * norm.cdf(-d1)
        )
    )

    return floorlet

#Ejemplo Caplet
N = 10_000_000

delta = 0.25

P = 0.98

F = 0.09

K = 0.08

sigma = 0.20

T = 0.25


caplet = caplet_black76(
    N,
    delta,
    P,
    F,
    K,
    sigma,
    T
)

print("Valor del Caplet:", caplet)

#Construcción de un Cap Completo
datos = pd.DataFrame({
    "Periodo": [1, 2, 3, 4, 5, 6, 7, 8],

    "T": [
        0.25, 0.50, 0.75, 1.00,
        1.25, 1.50, 1.75, 2.00
    ],

    "Forward": [
        0.090, 0.092, 0.095, 0.093,
        0.091, 0.094, 0.096, 0.098
    ],

    "Discount": [
        0.980, 0.960, 0.940, 0.920,
        0.900, 0.880, 0.860, 0.840
    ]
})

datos

#Cálculo de cada caplet
N = 10_000_000

K = 0.08

sigma = 0.20

delta = 0.25


datos["Caplet"] = datos.apply(
    lambda x: caplet_black76(
        N,
        delta,
        x["Discount"],
        x["Forward"],
        K,
        sigma,
        x["T"]
    ),
    axis=1
)

datos

#Valor del caplet
valor_cap = datos["Caplet"].sum()

print( f"Valor del CAP: ${valor_cap:,.2f}")

#Valuación de floor utilizando la misma tasa curva

datos["Floorlet"] = datos.apply(
    lambda x: floorlet_black76(
        N,
        delta,
        x["Discount"],
        x["Forward"],
        K,
        sigma,
        x["T"]
    ),
    axis=1
)

valor_floor = datos["Floorlet"].sum()

print(
    f"Valor del FLOOR: ${valor_floor:,.2f}"
)

#Construcción de un collar (Cap - floor)
collar = valor_cap - valor_floor

print( f"Valor del Collar: ${collar:,.2f}")

