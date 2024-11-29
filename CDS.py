import numpy as np
import matplotlib.pyplot as plt

# Parámetros del CDS
notional = 1000000  # Monto nocional
spread = 0.01       # Tasa del CDS (1%)
recovery_rate = 0.4 # Tasa de recuperación en caso de incumplimiento
default_prob = 0.02 # Probabilidad de incumplimiento anual
maturity = 5        # Vencimiento en años

# Calcula los flujos de efectivo anuales del CDS
premium_leg = notional * spread * np.ones(maturity)
protection_leg = notional * (1 - recovery_rate) * default_prob * np.arange(1, maturity + 1)

# Calcular el valor presente neto (NPV) de los flujos de efectivo
discount_factor = lambda t: 1 / ((1 + spread) ** t)
npv_premium_leg = np.sum(premium_leg * np.vectorize(discount_factor)(np.arange(1, maturity + 1)))
npv_protection_leg = np.sum(protection_leg * np.vectorize(discount_factor)(np.arange(1, maturity + 1)))

# Gráfico de los flujos de efectivo
plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, maturity + 1) - 0.2, premium_leg, width=0.4, label='Premium Leg', color='blue')
plt.bar(np.arange(1, maturity + 1) + 0.2, protection_leg, width=0.4, label='Protection Leg', color='red')
plt.xlabel('Years')
plt.ylabel('Cash Flows')
plt.title('Credit Default Swap Cash Flows')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar los resultados
print(f"NPV of Premium Leg: ${npv_premium_leg:.2f}")
print(f"NPV of Protection Leg: ${npv_protection_leg:.2f}")
