# ============================================================
# VALUACIÓN DE CAP Y FLOOR
# Modelo Black-76
# Ejercicio tipo DerivaGem
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# 1. DATOS DEL PROBLEMA
# ============================================================

N = 1_000_000          # Valor nominal
K = 0.045              # Tasa de ejercicio 4.5%
sigma = 0.01           # Volatilidad 1%
delta = 1.0            # Liquidaciones anuales

# Costo original del CAP:
costo_original = N * 0.005

# Tasas de mercado
tasas = {
    1: 0.0448,
    2: 0.0500,
    3: 0.0550,
    4: 0.0550
}

print("==============================================")
print("DATOS DEL CONTRATO")
print("==============================================")

print(f"Valor nominal: ${N:,.2f}")
print(f"Tasa de ejercicio: {K*100:.2f}%")
print(f"Volatilidad: {sigma*100:.2f}%")
print(f"Costo original CAP: ${costo_original:,.2f}")


# ============================================================
# 2. FACTORES DE DESCUENTO
# ============================================================

P = {}

for t, r in tasas.items():
    P[t] = 1 / ((1 + r) ** t)

print("\n==============================================")
print("FACTORES DE DESCUENTO")
print("==============================================")

for t in P:
    print(f"Año {t}: {P[t]:.8f}")


# ============================================================
# 3. TASAS FORWARD
# ============================================================

# Forward del primer periodo
F1 = (1 / P[1]) - 1

# Forward entre año 1 y año 2
F2 = (P[1] / P[2]) - 1

forwards = {
    1: F1,
    2: F2
}

print("\n==============================================")
print("TASAS FORWARD")
print("==============================================")

for t, f in forwards.items():
    print(f"Periodo {t}: {f*100:.6f}%")


# ============================================================
# 4. FUNCIÓN BLACK-76 PARA CAPLET
# ============================================================

def caplet_black76(F, K, sigma, T, delta, N, P):
    
    d1 = (
        np.log(F / K)
        + 0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    valor = (
        N * delta * P *
        (
            F * norm.cdf(d1)
            - K * norm.cdf(d2)
        )
    )
    
    return valor, d1, d2


# ============================================================
# 5. FUNCIÓN BLACK-76 PARA FLOORLET
# ============================================================

def floorlet_black76(F, K, sigma, T, delta, N, P):
    
    d1 = (
        np.log(F / K)
        + 0.5 * sigma**2 * T
    ) / (sigma * np.sqrt(T))
    
    d2 = d1 - sigma * np.sqrt(T)
    
    valor = (
        N * delta * P *
        (
            K * norm.cdf(-d2)
            - F * norm.cdf(-d1)
        )
    )
    
    return valor, d1, d2


# ============================================================
# 6. VALUACIÓN DEL CAP Y FLOOR
# ============================================================

resultados = []

for t in [1, 2]:

    F = forwards[t]
    descuento = P[t]

    cap, d1_cap, d2_cap = caplet_black76(
        F,
        K,
        sigma,
        t,
        delta,
        N,
        descuento
    )

    floor, d1_floor, d2_floor = floorlet_black76(
        F,
        K,
        sigma,
        t,
        delta,
        N,
        descuento
    )

    resultados.append([
        t,
        tasas[t] * 100,
        F * 100,
        descuento,
        d1_cap,
        d2_cap,
        cap,
        floor
    ])


# ============================================================
# 7. TABLA DE RESULTADOS
# ============================================================

df = pd.DataFrame(
    resultados,
    columns=[
        "Año",
        "Tasa mercado (%)",
        "Forward (%)",
        "Factor descuento",
        "d1 CAP",
        "d2 CAP",
        "CAPLET",
        "FLOORLET"
    ]
)

print("\n==============================================")
print("VALUACIÓN DE CAP Y FLOOR")
print("==============================================")

print(df.to_string(index=False))


# ============================================================
# 8. PRIMAS TOTALES
# ============================================================

prima_cap = df["CAPLET"].sum()
prima_floor = df["FLOORLET"].sum()

print("\n==============================================")
print("PRIMAS TOTALES")
print("==============================================")

print(f"Prima CAP:   ${prima_cap:,.2f}")
print(f"Prima FLOOR: ${prima_floor:,.2f}")


# ============================================================
# 9. COMPARACIÓN CON EL COSTO ORIGINAL
# ============================================================

ganancia_cap = prima_cap - costo_original

print("\n==============================================")
print("COMPARACIÓN CON COSTO ORIGINAL")
print("==============================================")

print(f"Costo original CAP: ${costo_original:,.2f}")
print(f"Valor actual CAP:   ${prima_cap:,.2f}")
print(f"Resultado:          ${ganancia_cap:,.2f}")

if ganancia_cap > 0:
    print("El CAP aumentó de valor.")
elif ganancia_cap < 0:
    print("El CAP disminuyó de valor.")
else:
    print("El CAP mantiene su valor original.")


# ============================================================
# 10. PAGOS DEL CAP Y FLOOR
# ============================================================

pagos = []

for t in [1, 2]:

    F = forwards[t]

    pago_cap = N * delta * max(F - K, 0)

    pago_floor = N * delta * max(K - F, 0)

    pagos.append([
        t,
        F * 100,
        pago_cap,
        pago_floor
    ])

df_pagos = pd.DataFrame(
    pagos,
    columns=[
        "Año",
        "Tasa Forward (%)",
        "Pago CAP",
        "Pago FLOOR"
    ]
)

print("\n==============================================")
print("PAGOS AL VENCIMIENTO")
print("==============================================")

print(df_pagos.to_string(index=False))


# ============================================================
# 11. GRÁFICA DEL PERFIL DE PAGOS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    df_pagos["Año"],
    df_pagos["Pago CAP"],
    marker="o",
    label="CAP"
)

plt.plot(
    df_pagos["Año"],
    df_pagos["Pago FLOOR"],
    marker="o",
    label="FLOOR"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Perfil de pagos del CAP y FLOOR")
plt.xlabel("Año")
plt.ylabel("Pago ($ MXN)")
plt.xticks([1, 2])
plt.grid(True)
plt.legend()

plt.show()
