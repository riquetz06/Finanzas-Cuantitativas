# ========================
# 3. CONDOR (calls)
# ========================
K1c, K2c, K3c, K4c = 80, 100, 120, 140
c1c, c2c, c3c, c4c = 18, 10, 6, 2   # primas ejemplo
premium_condor = c1c - c2c - c3c + c4c
condor = call_payoff(S, K1c) - call_payoff(S, K2c) - call_payoff(S, K3c) + call_payoff(S, K4c) - premium_condor

plt.figure(figsize=(8,5))
plt.plot(S, condor, label='Condor (Long)', color="brown", linewidth=2)
plt.axhline(0, color='black', linestyle='--')
plt.axvline(K1c, color='orange', linestyle='--', label=f'K1={K1c}')
plt.axvline(K2c, color='red', linestyle='--', label=f'K2={K2c}')
plt.axvline(K3c, color='blue', linestyle='--', label=f'K3={K3c}')
plt.axvline(K4c, color='purple', linestyle='--', label=f'K4={K4c}')
plt.title("Estrategia Condor (Long)")
plt.xlabel("Precio del subyacente ($S_T$)")
plt.ylabel("Ganancia / Pérdida")
plt.legend()
plt.grid(True)
plt.show()
