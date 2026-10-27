pip install pandas numpy scipy 
matplotlib openpyxl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# =====================================================
# 1. PARÁMETROS
# =====================================================

ARCHIVO = "rendimientos.xlsx"
HOJA = "Rendimientos"

PERIODOS_ANUALES = 252
TASA_LIBRE_RIESGO = 0.08  # 8% anual; modificar según corresponda

# Límites de inversión
PESO_MINIMO = 0.00
PESO_MAXIMO = 1.00

# =====================================================
# 2. CARGAR RENDIMIENTOS DESDE EXCEL
# =====================================================

datos = pd.read_excel(ARCHIVO, sheet_name=HOJA)

# La primera columna puede contener fechas.
# Se conservan las columnas numéricas como activos.
rendimientos = datos.select_dtypes(include="number").copy()

# Eliminar columnas totalmente vacías
rendimientos = rendimientos.dropna(axis=1, how="all")

# Eliminar filas incompletas para usar la misma muestra
rendimientos = rendimientos.dropna(axis=0, how="any")

if rendimientos.empty:
    raise ValueError("No hay rendimientos numéricos completos.")

if rendimientos.shape[1] < 2:
    raise ValueError("Se necesitan al menos dos activos.")

if len(rendimientos) < 2:
    raise ValueError("Se necesitan al menos dos observaciones.")

if not np.isfinite(rendimientos.to_numpy()).all():
    raise ValueError("Existen valores infinitos o no válidos.")

activos = rendimientos.columns.tolist()
n = len(activos)

# =====================================================
# 3. ESTIMACIÓN DE PARÁMETROS
# =====================================================

# Rendimientos logarítmicos medios anualizados
mu = rendimientos.mean().to_numpy() * PERIODOS_ANUALES

# Matriz de covarianzas anualizada
cov = rendimientos.cov().to_numpy() * PERIODOS_ANUALES

# Corregir pequeñas asimetrías numéricas
cov = (cov + cov.T) / 2

# =====================================================
# 4. FUNCIONES DE LA CARTERA
# =====================================================

def rendimiento_cartera(pesos):
    return float(pesos @ mu)


def varianza_cartera(pesos):
    return float(pesos @ cov @ pesos)


def volatilidad_cartera(pesos):
    return np.sqrt(max(varianza_cartera(pesos), 0))


def sharpe_cartera(pesos):
    volatilidad = volatilidad_cartera(pesos)

    if volatilidad <= 1e-12:
        return -1e10

    return (
        rendimiento_cartera(pesos) - TASA_LIBRE_RIESGO
    ) / volatilidad


# =====================================================
# 5. RESTRICCIONES Y LÍMITES
# =====================================================

restricciones = [
    {
        "type": "eq",
        "fun": lambda pesos: np.sum(pesos) - 1
    }
]

limites = [
    (PESO_MINIMO, PESO_MAXIMO)
    for _ in range(n)
]

pesos_iniciales = np.ones(n) / n


# =====================================================
# 6. CARTERA DE MÍNIMA VARIANZA
# =====================================================

resultado_min = minimize(
    fun=varianza_cartera,
    x0=pesos_iniciales,
    method="SLSQP",
    bounds=limites,
    constraints=restricciones,
    options={"maxiter": 2000, "ftol": 1e-12}
)

if not resultado_min.success:
    raise RuntimeError(
        "Falló la optimización de mínima varianza: "
        + resultado_min.message
    )

pesos_min = resultado_min.x


# =====================================================
# 7. CARTERA DE MÁXIMO SHARPE
# =====================================================

resultado_sharpe = minimize(
    fun=lambda pesos: -sharpe_cartera(pesos),
    x0=pesos_iniciales,
    method="SLSQP",
    bounds=limites,
    constraints=restricciones,
    options={"maxiter": 2000, "ftol": 1e-12}
)

if not resultado_sharpe.success:
    raise RuntimeError(
        "Falló la optimización de Sharpe: "
        + resultado_sharpe.message
    )

pesos_sharpe = resultado_sharpe.x


# =====================================================
# 8. RESUMEN DE RESULTADOS
# =====================================================

def metricas(pesos):
    return {
        "Rendimiento logarítmico anualizado":
            rendimiento_cartera(pesos),
        "Volatilidad anualizada":
            volatilidad_cartera(pesos),
        "Varianza anualizada":
            varianza_cartera(pesos),
        "Sharpe":
            sharpe_cartera(pesos)
    }


resumen = pd.DataFrame(
    {
        "Mínima varianza": metricas(pesos_min),
        "Máximo Sharpe": metricas(pesos_sharpe)
    }
).T

ponderaciones = pd.DataFrame(
    {
        "Activo": activos,
        "Mínima varianza": pesos_min,
        "Máximo Sharpe": pesos_sharpe
    }
)

print("\n===== PONDERACIONES ÓPTIMAS =====")
print(
    ponderaciones.to_string(
        index=False,
        formatters={
            "Mínima varianza": "{:.4%}".format,
            "Máximo Sharpe": "{:.4%}".format
        }
    )
)

print("\n===== MÉTRICAS DE LAS CARTERAS =====")
print(resumen.round(6))


# =====================================================
# 9. FRONTERA EFICIENTE
# =====================================================

# Rangos factibles de rendimiento logarítmico esperado
# bajo las restricciones de pesos y sin ventas en corto.
rendimientos_objetivo = np.linspace(
    mu.min(), mu.max(), 60
)

frontera = []

for objetivo in rendimientos_objetivo:

    restricciones_frontera = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        },
        {
            "type": "eq",
            "fun": lambda w, r=objetivo:
                rendimiento_cartera(w) - r
        }
    ]

    resultado = minimize(
        fun=varianza_cartera,
        x0=pesos_iniciales,
        method="SLSQP",
        bounds=limites,
        constraints=restricciones_frontera,
        options={"maxiter": 2000, "ftol": 1e-11}
    )

    if resultado.success:
        frontera.append({
            "Rendimiento": rendimiento_cartera(resultado.x),
            "Volatilidad": volatilidad_cartera(resultado.x)
        })

frontera = pd.DataFrame(frontera)


# =====================================================
# 10. GRÁFICA DE LA FRONTERA EFICIENTE
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    frontera["Volatilidad"] * 100,
    frontera["Rendimiento"] * 100,
    label="Frontera eficiente"
)

plt.scatter(
    volatilidad_cartera(pesos_min) * 100,
    rendimiento_cartera(pesos_min) * 100,
    marker="o",
    s=100,
    label="Mínima varianza"
)

plt.scatter(
    volatilidad_cartera(pesos_sharpe) * 100,
    rendimiento_cartera(pesos_sharpe) * 100,
    marker="*",
    s=180,
    label="Máximo Sharpe"
)

plt.xlabel("Volatilidad anualizada (%)")
plt.ylabel("Rendimiento logarítmico anualizado (%)")
plt.title("Optimización de cartera de inversión")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =====================================================
# 11. EXPORTAR RESULTADOS A EXCEL
# =====================================================

with pd.ExcelWriter("resultados_optimizacion.xlsx") as writer:

    ponderaciones.to_excel(
        writer, sheet_name="Ponderaciones", index=False
    )

    resumen.to_excel(
        writer, sheet_name="Metricas"
    )

    frontera.to_excel(
        writer, sheet_name="Frontera", index=False
    )

    pd.DataFrame(
        mu, index=activos, columns=["Rendimiento log anualizado"]
    ).to_excel(writer, sheet_name="Rendimientos")

    pd.DataFrame(
        cov, index=activos, columns=activos
    ).to_excel(writer, sheet_name="Covarianzas")

print("\nArchivo generado: resultados_optimizacion.xlsx")
