import pandas as pd
import numpy as np

# =========================
# CONFIGURACIÓN
# =========================

N = 60000  # 5 días de operación (~12.000/día)
np.random.seed(42)  # reproducibilidad

# =========================
# GENERACIÓN DE DATOS
# =========================

data = []
for i in range(N):

    # -------------------------
    # 1. DÍAS DE IMPAGO (1-90)
    # -------------------------
    dias = np.random.randint(1, 91)

    # 2. IMPORTE (1.000 - 30.000)

    importe = np.random.lognormal(mean=9.5, sigma=0.6)
    importe = max(1000, min(30000, importe))

    # 3. PRIORIDAD BASE (1-5)
    prioridad = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.30, 0.20, 0.10])

    # 4. HISTORIAL (0-5)

    if dias <= 30:
        historial = np.random.choice([0,1,2], p=[0.6, 0.3, 0.1])
    elif dias <= 60:
        historial = np.random.choice([1,2,3,4], p=[0.2, 0.3, 0.3, 0.2])
    else:
        historial = np.random.choice([2,3,4,5], p=[0.1, 0.2, 0.4, 0.3])

    # 5. REACTIVACIÓN

    if dias > 60 and historial >= 3:
        reactivacion = np.random.choice([0, 1], p=[0.3, 0.7])
    elif dias > 30 and historial >= 2:
        reactivacion = np.random.choice([0, 1], p=[0.6, 0.4])
    else:
        reactivacion = np.random.choice([0, 1], p=[0.9, 0.1])
    
    # ID
    expediente_id = f"EXP_{i:06d}"

    # GUARDAR
    data.append([
        expediente_id,
        dias,
        round(importe, 2),
        prioridad,
        historial,
        reactivacion
    ])

df = pd.DataFrame(data, columns=[
    "expediente_id",
    "dias_impago",
    "importe",
    "prioridad_base",
    "historial",
    "reactivacion"
])

# EXPORTAR CSV

df.to_csv("expedientes_gyar.csv", index=False)
print("CSV generado correctamente: 60.000 expedientes")