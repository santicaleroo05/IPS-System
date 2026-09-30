import pandas as pd
import numpy as np

# =========================
# 1. CARGA DE DATOS
# =========================

df = pd.read_csv("expedientes_gyar.csv")

# =========================
# 2. FEATURE ENGINEERING
# =========================

# -------------------------
# TRAMO OPERATIVO
# -------------------------
def calcular_tramo(dias):
    if dias <= 30:
        return "0-30"
    elif dias <= 60:
        return "31-60"
    else:
        return "61-90"

df["tramo"] = df["dias_impago"].apply(calcular_tramo)

# =========================
# 3. SCORING VARIABLES
# =========================

# -------------------------
# DÍAS
# -------------------------
def score_dias(dias):
    if dias <= 30:
        return 1
    elif dias <= 60:
        return 3
    else:
        return 5

df["dias_score"] = df["dias_impago"].apply(score_dias)

# -------------------------
# IMPORTE (ADAPTATIVO)
# -------------------------
p33_imp = df["importe"].quantile(0.33)
p66_imp = df["importe"].quantile(0.66)

def score_importe(x):
    if x <= p33_imp:
        return 1
    elif x <= p66_imp:
        return 3
    else:
        return 5

df["importe_score"] = df["importe"].apply(score_importe)

# -------------------------
# PRIORIDAD BASE
# -------------------------
df["prioridad_score"] = df["prioridad_base"]

# -------------------------
# HISTORIAL
# -------------------------
df["historial_score"] = df["historial"]

# =========================
# 4. MOTOR DE DECISIÓN
# =========================

# -------------------------
# PRIORIDAD FINAL
# -------------------------
df["prioridad_final"] = (
    0.40 * df["dias_score"] +
    0.25 * df["prioridad_score"] +
    0.20 * df["importe_score"] +
    0.15 * df["historial_score"]
)

# -------------------------
# ESFUERZO OPERATIVO
# -------------------------
df["esfuerzo"] = (
    0.50 * df["historial_score"] +
    0.30 * df["dias_score"] +
    0.20 * df["prioridad_score"]
)

# -------------------------
# REDONDEO
# -------------------------
df["prioridad_final"] = df["prioridad_final"].round(2)
df["esfuerzo"] = df["esfuerzo"].round(2)

# -------------------------
# VALOR OPERATIVO
# -------------------------
df["valor_operativo"] = (
    df["prioridad_final"] / df["esfuerzo"]
).round(2)

# =========================
# 5. PRIORIZACIÓN Y SECUENCIACIÓN
# =========================

# Orden inteligente de trabajo
df = df.sort_values(
    "valor_operativo",
    ascending=False
).reset_index(drop=True)

# Cola operativa
df["orden_gestion"] = range(1, len(df) + 1)

# =========================
# 6. ASIGNACIÓN A GESTORES
# =========================

n_gestores = 39

gestores = [f"G{i}" for i in range(n_gestores)]

# carga real acumulada
carga_gestores = {g: 0 for g in gestores}

asignaciones = []

for _, row in df.iterrows():

    # gestor con menor carga
    gestor_min = min(
        carga_gestores,
        key=carga_gestores.get
    )

    asignaciones.append(gestor_min)

    # equilibrio basado en esfuerzo REAL
    carga_gestores[gestor_min] += row["esfuerzo"]

df["gestor_asignado"] = asignaciones

# =========================
# 7. KPIs DEL SISTEMA
# =========================

# -------------------------
# CARGA POR GESTOR
# -------------------------
cargas = df.groupby(
    "gestor_asignado"
)["esfuerzo"].sum()

kpi_carga_media = cargas.mean()

kpi_desviacion_carga = cargas.std()

# -------------------------
# EXPEDIENTES CRÍTICOS
# -------------------------
criticos = df[
    df["prioridad_final"] >= 4
]

# -------------------------
# CAPACIDAD OPERATIVA
# Dataset = 5 días simulados
# -------------------------
capacidad_total = (
    n_gestores * 300 * 5
)

# -------------------------
# KPI INTERVENCIÓN TEMPRANA
# -------------------------
gestion_temprana = df[
    (df["dias_impago"] <= 60) &
    (df["orden_gestion"] <= capacidad_total)
]

kpi_intervencion_temprana = (
    len(gestion_temprana) / len(df)
) * 100

# -------------------------
# PRODUCTIVIDAD
# -------------------------
dias_simulados = 5

kpi_productividad = (
    len(df) /
    n_gestores /
    dias_simulados
)

# =========================
# 8. BASELINE ALEATORIO
# =========================

df_random = df.copy()

np.random.seed(42)

# rompe secuencia
df_random = df_random.sample(
    frac=1
).reset_index(drop=True)

# asignación aleatoria
df_random["gestor_asignado"] = np.random.choice(
    gestores,
    size=len(df_random)
)

# -------------------------
# KPIs RANDOM
# -------------------------
carga_random = df_random.groupby(
    "gestor_asignado"
)["esfuerzo"].sum()

kpi_random_std = carga_random.std()

# =========================
# 9. COMPARACIÓN
# =========================

mejora_carga = (
    (
        kpi_random_std -
        kpi_desviacion_carga
    )
    / kpi_random_std
) * 100

# =========================
# 10. OUTPUT
# =========================

print("")
print("===== SISTEMA IPS EJECUTADO =====")

print(f"Carga media por gestor: {kpi_carga_media:.2f}")
print(
    f"Desviación de carga: "
    f"{kpi_desviacion_carga:.2f}"
)
print(
    f"Expedientes críticos: "
    f"{len(criticos)}"
)
print(
    f"% intervención temprana: "
    f"{kpi_intervencion_temprana:.2f}%"
)

print(
    f"Productividad media expedientes/gestor: "
    f"{kpi_productividad:.2f}"
)

print("==================================")

print("")

print("===== COMPARACIÓN IPS vs ALEATORIO =====")

print(
    f"Desviación carga (IPS): "
    f"{kpi_desviacion_carga:.2f}"
)

print(
    f"Desviación carga (Random): "
    f"{kpi_random_std:.2f}"
)

print("")

print(
    f"Mejora en equilibrio de carga: "
    f"{mejora_carga:.2f}%"
)

print("========================================")

# =========================
# 11. EXPORTS
# =========================

df.to_csv(
    "ips_modelo.csv",
    index=False
)

df_random.to_csv(
    "baseline_aleatorio.csv",
    index=False
)