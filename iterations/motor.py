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
# DÍAS (CORRECTO: OPERATIVO, NO ESTADÍSTICO)
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
# IMPORTE (percentiles → adaptativo diario)
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
# PRIORIDAD (ya viene del CSV)
# -------------------------
df["prioridad_score"] = df["prioridad_base"]

# -------------------------
# HISTORIAL (ya viene del CSV)
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
# VALOR OPERATIVO (OPTIMIZACIÓN)
# -------------------------
df["valor_operativo"] = df["prioridad_final"] / df["esfuerzo"]

# =========================
# 5. ASIGNACIÓN A GESTORES
# =========================

n_gestores = 39
gestores = [f"G{i}" for i in range(n_gestores)]

carga_gestores = {g: 0 for g in gestores}

df = df.sort_values("valor_operativo", ascending=False)

asignaciones = []

for _, row in df.iterrows():
    gestor_min = min(carga_gestores, key=carga_gestores.get)
    
    asignaciones.append(gestor_min)
    carga_gestores[gestor_min] += row["esfuerzo"]

df["gestor_asignado"] = asignaciones

# =========================
# 6. KPIs DEL SISTEMA
# =========================

# carga por gestor
cargas = df.groupby("gestor_asignado")["esfuerzo"].sum()

kpi_carga_media = cargas.mean()
kpi_desviacion_carga = cargas.std()

# expedientes críticos
criticos = df[df["prioridad_final"] >= 4]

# KPI operativo (<60 días)
df["gestion_antes_60"] = df["dias_impago"] < 60
kpi_60 = df["gestion_antes_60"].mean()

# =========================
# 7. EXPORT FINAL
# =========================

df.to_csv("resultado_ips.csv", index=False)

# =========================
# 8. OUTPUT RESUMEN
# =========================

print("===== SISTEMA IPS EJECUTADO =====")
print(f"Carga media por gestor: {kpi_carga_media:.2f}")
print(f"Desviación de carga: {kpi_desviacion_carga:.2f}")
print(f"Expedientes críticos: {len(criticos)}")
print(f"% de cartera dentro de ventana de intervención temprana: {kpi_60*100:.2f}%")
print("==================================")