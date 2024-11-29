import numpy as np
import matplotlib.pyplot as plt

# Definir los parámetros de la opción
precio_base = 100  # Precio del activo subyacente
prima = 5          # Prima pagada por la opción
strike = 105       # Precio de ejercicio

# Definir una función para calcular los beneficios
def beneficio_long_call(precio_final):
    return np.maximum(precio_final - strike, 0) - prima

# Generar una secuencia de precios finales
precios_finales = np.linspace(80, 130, 100)

# Calcular los beneficios
beneficios = beneficio_long_call(precios_finales)

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(precios_finales, beneficios, label='Beneficio', color='blue')
plt.axhline(0, color='red', linestyle='dashed')
plt.axvline(strike, color='green', linestyle='dashed')
plt.title('Gráfica de Long Call')
plt.xlabel('Precio Final del Activo Subyacente')
plt.ylabel('Beneficio')
plt.legend()
plt.grid(True)
plt.show()
