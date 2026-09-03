import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# DATOS DEL PRÉSTAMO
# ============================================================

N = 5_000_000       # Valor del crédito
K = 0.055            # Tasa strike = 5.50%
sigma = 0.15         # Volatilidad = 15%
delta = 0.5          # Periodicidad semestral
r = 0.055            # Tasa de descuento plana

# Tasas forward proporcionadas
meses = np.array([6, 12, 18, 24, 30, 36])

forward = np.array([
    0.0304,   # 3.04%
    0.0464,   # 4.64%
    0.0476,   # 4.76%
    0.0539,   # 5.39%
    0.0578,   # 5.78%
    0.0558    # 5.58%
])

# ============================================================
# FUNCIÓN BLACK-76 PARA CAPLET
# ============================================================

def caplet_black76(F, K, sigma, T, delta, N, discount):
    
    # Caso T = 0
    if T == 0:
        return N * delta * discount * max(F - K, 0)
    
    d1 = (
        np.log(F / K) +
        0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    precio = (
        N * delta * discount *
        (F * norm.cdf(d1) -
         K * norm.cdf(d2))
    )
    
    return precio


# ============================================================
# FUNCIÓN BLACK-76 PARA FLOORLET
# ============================================================

def floorlet_black76(F, K, sigma, T, delta, N, discount):
    
    # Caso T = 0
    if T == 0:
        return N * delta * discount * max(K - F, 0)
    
    d1 = (
        np.log(F / K) +
        0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    precio = (
        N * delta * discount *
        (K * norm.cdf(-d2) -
         F * norm.cdf(-d1))
    )
    
    return precio


# ============================================================
# CÁLCULO
# ============================================================

resultados = []

for m, F in zip(meses, forward):
    
    # Tiempo hasta la fijación de la tasa
    T = (m - 6) / 12
    
    # Tiempo hasta el pago
    payment = m / 12
    
    # Factor de descuento
    discount = np.exp(-r * payment)
    
    # Caplet
    caplet = caplet_black76(
        F, K, sigma, T, delta, N, discount
    )
    
    # Floorlet
    floorlet = floorlet_black76(
        F, K, sigma, T, delta, N, discount
    )
    
    resultados.append([
        m,
        F * 100,
        T,
        payment,
        discount,
        caplet,
        floorlet
    ])


# ============================================================
# TABLA DE RESULTADOS
# ============================================================

df = pd.DataFrame(
    resultados,
    columns=[
        "Plazo (meses)",
        "Tasa Forward (%)",
        "T fijación",
        "T pago",
        "Factor descuento",
        "Caplet",
        "Floorlet"
    ]
)

# Mostrar resultados
pd.set_option("display.float_format", "{:,.2f}".format)

print(df)

# ============================================================
# PRIMAS TOTALES
# ============================================================

prima_cap = df["Caplet"].sum()
prima_floor = df["Floorlet"].sum()

print("\n======================================")
print("PRIMAS TOTALES")
print("======================================")

print(f"Prima total CAP   : ${prima_cap:,.2f}")
print(f"Prima total FLOOR : ${prima_floor:,.2f}")

# Porcentaje respecto al crédito

print("\n======================================")
print("PRIMA COMO % DEL CRÉDITO")
print("======================================")

print(f"CAP   : {prima_cap/N*100:.4f}%")
print(f"FLOOR : {prima_floor/N*100:.4f}%")
