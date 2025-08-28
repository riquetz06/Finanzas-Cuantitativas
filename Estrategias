import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Funciones de payoff
# ==========================
def call_payoff(ST, K):
    return np.maximum(ST - K, 0.0)

def put_payoff(ST, K):
    return np.maximum(K - ST, 0.0)

# ==========================
# Parámetros
# ==========================
S = np.linspace(50, 200, 500)   # rango de precios del subyacente

# Bull Call Spread
K1_bull, K2_bull = 100, 120
c_long, c_short = 10, 4
bull = (call_payoff(S, K1_bull) - c_long) - (call_payoff(S, K2_bull) - c_short)

# Bear Put Spread
K1_bear, K2_bear = 100, 120
p_long, p_short = 15, 7
bear = (put_payoff(S, K2_bear) - p_long) - (put_payoff(S, K1_bear) - p_short)

# Straddle (Call + Put al mismo strike)
K_straddle = 110
c_straddle, p_straddle = 9, 8
straddle = call_payoff(S, K_straddle) + put_payoff(S, K_straddle) - (c_straddle + p_straddle)

# Strangle (Call + Put con strikes distintos)
Kp, Kc = 100, 120
c_strangle, p_strangle = 7, 6
strangle = call_payoff(S, Kc) + put_payoff(S, Kp) - (c_strangle + p_strangle)

# ==========================
# Gráficas
# ==========================
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot(S, bull, label="Bull Call Spread")
plt.axhline(0, color="black", linestyle="--")
plt.title("Bull Call Spread")
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(S, bear, label="Bear Put Spread", color="red")
plt.axhline(0, color="black", linestyle="--")
plt.title("Bear Put Spread")
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(S, straddle, label="Straddle", color="green")
plt.axhline(0, color="black", linestyle="--")
plt.title("Straddle")
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(S, strangle, label="Strangle", color="purple")
plt.axhline(0, color="black", linestyle="--")
plt.title("Strangle")
plt.grid(True)

plt.tight_layout()
plt.show()
