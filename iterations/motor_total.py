import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. CARGA DE DATOS

df = pd.read_csv("expedientes_gyar.csv")

# 2. FEATURE ENGINEERING

# TRAMO OPERATIVO

def calcular_tramo(dias):
    if dias <= 30:
        return "0-30"
    elif dias <= 60:
        return "31-60"
    else:
        return "61-90"

df["tramo"] = df["dias_impago"].apply(calcular_tramo)

# 3. SCORING VARIABLES

# IMPORTE

p33 = df["importe"].quantile(0.33) # El 33% de las deudas más bajas
p66 = df["importe"].quantile(0.66) # El 66% (límite para deudas altas)

imp_min = df["importe"].min()
imp_max = df["importe"].max()

df["importe_score"] = (1 + 4 * (
    (df["importe"] - imp_min) /
    (imp_max - imp_min)
)).round(2)

# PRIORIDAD BASE

df["prioridad_score"] = df["prioridad_base"]

# HISTORIAL

df["historial_score"] = df["historial"]

# DÍAS (score continuo)

dias_min = df["dias_impago"].min()
dias_max = df["dias_impago"].max()

df["dias_score"] = (1 + 4 * (
    (df["dias_impago"] - dias_min) /
    (dias_max - dias_min)
)).round(2)

# BOOST REACTIVACIÓN

df["boost_reactivacion"] = df["reactivacion"].apply(
    lambda x: 0.5 if x == 1 else 0
)

# 4. MOTOR DE DECISIÓN

# PRIORIDAD FINAL

df["prioridad_final"] = (
    0.40 * df["dias_score"] +
    0.25 * df["prioridad_score"] +
    0.20 * df["importe_score"] +
    0.15 * df["historial_score"]+
    df["boost_reactivacion"]
).round(2)

# ESFUERZO OPERATIVO

df["esfuerzo"] = (
    0.50 * df["historial_score"] +
    0.30 * df["dias_score"] +
    0.20 * df["prioridad_score"]
).round(2)

# VALOR OPERATIVO

df["valor_operativo"] = (
    df["prioridad_final"] / df["esfuerzo"]
).round(2)

# 5. PRIORIZACIÓN Y SECUENCIACIÓN

# Orden inteligente de trabajo
df = df.sort_values(
    "valor_operativo",
    ascending=False
).reset_index(drop=True)

# Cola operativa
df["orden_gestion"] = range(1, len(df) + 1)

# 6. ASIGNACIÓN A GESTORES

n_gestores = 39
# 1. Definimos los rangos
gestores_junior = [f"G_JR_{i}" for i in range(1, 14)]
gestores_adv = [f"G_ADV_{i}" for i in range(1, 14)]
gestores_expert = [f"G_EXP_{i}" for i in range(1, 14)]
gestores = gestores_junior + gestores_adv + gestores_expert

carga_gestores = {g: 0 for g in gestores}
asignaciones = []

for _, row in df.iterrows():
    # Determinamos la complejidad del caso por Percentiles e Impago
    es_critico = row["dias_impago"] > 60 or row["importe"] > p66
    es_intermedio = 30 < row["dias_impago"] <= 60 or p33 < row["importe"] <= p66

    if es_critico:
        pool_ideal = gestores_expert
    elif es_intermedio:
        pool_ideal = gestores_adv
    else:
        pool_ideal = gestores_junior

    # ESTRATEGIA DE EQUIDAD TOTAL:
    # Primero buscamos al gestor con menos carga de TODA la plataforma
    gestor_min_global = min(carga_gestores, key=carga_gestores.get)
    carga_minima = carga_gestores[gestor_min_global]

    # Para no generar desigualdad, solo permitimos elegir por "afinidad" 
    # si el gestor ideal no está mucho más cargado que el mínimo global.
    # Umbral de tolerancia (ej. 100 unidades de esfuerzo)
    tolerancia = 100 
    
    # Filtramos gestores del rango ideal que están dentro del rango de equidad
    candidatos_afines = [g for g in pool_ideal if carga_gestores[g] <= carga_minima + tolerancia]

    if candidatos_afines:
        # Si hay gente en su rango ideal disponible y no saturada, elegimos al de menos carga
        gestor_final = min(candidatos_afines, key=lambda g: carga_gestores[g])
    else:
        # Si todo su rango ideal está saturado, se asigna al mínimo global (desbordamiento)
        gestor_final = gestor_min_global

    asignaciones.append(gestor_final)
    carga_gestores[gestor_final] += row["esfuerzo"]

df["gestor_asignado"] = asignaciones
df["perfil_gestor"] = df["gestor_asignado"].apply(lambda x: x.split('_')[1])

# 7. KPIs DEL SISTEMA

# CARGA POR GESTOR

cargas = df.groupby("gestor_asignado")["esfuerzo"].sum()

kpi_carga_media = cargas.mean()
kpi_desviacion_carga = cargas.std()

# EXPEDIENTES CRÍTICOS

criticos = df[ df["prioridad_final"] >= 4]

# CAPACIDAD OPERATIVA
# Dataset = 5 días simulados

capacidad_total = (n_gestores * 300 * 5)

# KPI INTERVENCIÓN TEMPRANA

gestion_temprana = df[ (df["dias_impago"] <= 60) & (df["orden_gestion"] <= capacidad_total)]
kpi_intervencion_temprana = (len(gestion_temprana) / len(df)) * 100

# PRODUCTIVIDAD
dias_simulados = 5
kpi_productividad = (
    len(df) /
    n_gestores /
    dias_simulados
)
prod_por_gestor = df.groupby("gestor_asignado").size()
prod_por_gestor = prod_por_gestor / dias_simulados

# ESPECIALIZACIÓN
# Porcentaje de casos críticos atendidos por expertos
expertos_en_criticos = df[(df["prioridad_final"] >= 4) & (df["perfil_gestor"] == "EXP")]
ratio_afinidad = (len(expertos_en_criticos) / len(df[df["prioridad_final"] >= 4])) * 100

# 8. BASELINE ALEATORIO

df_random = df.copy()
np.random.seed(42)

# rompe secuencia
df_random = df_random.sample(frac=1).reset_index(drop=True)

# asignación aleatoria
df_random["gestor_asignado"] = np.random.choice( gestores,size=len(df_random))

# KPIs RANDOM

carga_random = df_random.groupby( "gestor_asignado")["esfuerzo"].sum()
kpi_random_std = carga_random.std()

# 9. COMPARACIÓN

mejora_carga = (
    (
        kpi_random_std -
        kpi_desviacion_carga
    )
    / kpi_random_std
) * 100

# 10. OUTPUT

print("")
print("===== SISTEMA IPS EJECUTADO =====")

print(f"Carga media por gestor: {kpi_carga_media:.2f}")
print( f"Desviación de carga: " f"{kpi_desviacion_carga:.2f}")
print(f"Expedientes críticos: " f"{len(criticos)}")
print(f"% intervención temprana: " f"{kpi_intervencion_temprana:.2f}%")
print(f"Productividad media expedientes/gestor: " f"{kpi_productividad:.2f}")
print(f"Índice de Afinidad (Casos Críticos -> Expertos): {ratio_afinidad:.2f}%")

print("==================================")

print("")

print("===== COMPARACIÓN IPS vs ALEATORIO =====")

print(f"Desviación carga (IPS): " f"{kpi_desviacion_carga:.2f}")
print(f"Desviación carga (Random): " f"{kpi_random_std:.2f}")

print("")

print(f"Mejora en equilibrio de carga: " f"{mejora_carga:.2f}%")

print("========================================")

# 11. EXPORTS

df.to_csv("ips_modelo.csv", index=False)
df_random.to_csv("baseline_aleatorio.csv", index=False)

# CARGA POR GESTOR (IPS)

carga_ips = df.groupby("gestor_asignado")["esfuerzo"].sum()

# CARGA POR GESTOR (RANDOM)

carga_random = df_random.groupby("gestor_asignado")["esfuerzo"].sum()

# DATAFRAME COMPARATIVO

df_plot = pd.DataFrame({
    "IPS": carga_ips,
    "Random": carga_random
}).fillna(0)

# 12. PLOT

plt.figure(figsize=(14,6))

df_plot.sort_values("IPS", inplace=True)

plt.plot(df_plot.index, df_plot["IPS"], label="IPS (modelo)", linewidth=2)
plt.plot(df_plot.index, df_plot["Random"], label="Random", linewidth=2, alpha=0.7)

plt.title("Comparación de carga operativa por gestor")
plt.xlabel("Gestor")
plt.ylabel("Carga total (esfuerzo)")
plt.xticks(rotation=90)
plt.legend()

plt.tight_layout()
plt.savefig("comparacion_ips_vs_random.png", dpi=300)
plt.show()

# =========================
plt.figure(figsize=(6,4))

plt.bar(["IPS", "Random"], [kpi_desviacion_carga, kpi_random_std])

plt.title("Desviación de carga operativa")
plt.ylabel("Desviación estándar")

plt.savefig("kpi_desviacion.png", dpi=300)
plt.show()

# =========================
plt.figure(figsize=(14,6))

prod_por_gestor.sort_index().plot(kind="bar")

plt.title("Productividad por gestor (expedientes/día)")
plt.xlabel("Gestor")
plt.ylabel("Expedientes por día")

plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("kpi_productividad_por_gestor.png", dpi=300)
plt.show()