import streamlit as st
import pandas as pd
import plotly.express as px

# EJECUCIÓN
"""
Ejecutar en la terminal: streamlit run app_predict.py
"""

# CONFIGURACIÓN

st.set_page_config(page_title="IPS Dashboard", layout="wide")
st.title("Sistema IPS - Optimización de Asignación Operativa")

# CARGA DE DATOS

df = pd.read_csv("ips_modelo.csv")
df_random = pd.read_csv("baseline_aleatorio.csv")
n_gestores=39
capacidad_total = (n_gestores * 300 * 5)

# KPIs GLOBALES

cargas_ips = df.groupby("gestor_asignado")["esfuerzo"].sum()
cargas_random = df_random.groupby("gestor_asignado")["esfuerzo"].sum()

kpi_std_ips = cargas_ips.std()
kpi_std_random = cargas_random.std()
kpi_productividad = len(df) / 39 / 5
gestion_temprana = df[ (df["dias_impago"] <= 60) & (df["orden_gestion"] <= capacidad_total)]
kpi_intervencion_temprana = (len(gestion_temprana) / len(df)) * 100

# DASHBOARD KPI

col1, col2, col3 = st.columns(3)

col1.metric("Desviación IPS", f"{kpi_std_ips:.2f}")
col2.metric("Desviación Random", f"{kpi_std_random:.2f}")
col3.metric("Mejora equilibrio (%)", f"{((kpi_std_random - kpi_std_ips)/kpi_std_random)*100:.2f}%")

col4, col5 = st.columns(2)

col4.metric("Intervención temprana (%)", f"{kpi_intervencion_temprana:.2f}")
col5.metric("Productividad exp/gestor/día", f"{kpi_productividad:.2f}")

st.divider()

# COMPARACIÓN CARGA GESTORES

st.subheader("Comparación de carga por gestor")

df_plot = pd.DataFrame({"IPS": cargas_ips,"Random": cargas_random}).fillna(0)

fig = px.line(
    df_plot,
    x=df_plot.index,
    y=["IPS", "Random"],
    title="Carga operativa por gestor",
    labels={"value": "Carga", "index": "Gestor"}
)

st.plotly_chart(fig, use_container_width=True)

# DISTRIBUCIÓN

st.subheader("Distribución de carga")

fig2 = px.box(
    df,
    x="gestor_asignado",
    y="esfuerzo",
    title="Distribución de esfuerzo por gestor (IPS)"
)

st.plotly_chart(fig2, use_container_width=True)

# TOP EXPEDIENTES

st.subheader("Top expedientes prioritarios")
top = df.sort_values("prioridad_final", ascending=False).head(20)

st.dataframe(top[[
    "dias_impago",
    "importe",
    "prioridad_final",
    "esfuerzo",
    "gestor_asignado"
]])

# HISTOGRAMA PRIORIDAD

st.subheader("Distribución de prioridad")

fig3 = px.histogram(
    df,
    x="prioridad_final",
    nbins=30,
    title="Distribución de prioridad operativa"
)

st.plotly_chart(fig3, use_container_width=True)

# VALIDACIÓN DEL MODELO ML

st.subheader("Validación del modelo predictivo")

df["es_critico"] = df["prioridad_base"] >= 4
df["rank_operativo"] = df["valor_operativo"].rank(ascending=False)
df["rank_reglas"] = df["prioridad_final"].rank(ascending=False)
df["rank_risk"] = df["risk_score"].rank(ascending=False)

criticos = df[df["es_critico"]]

metricas_ml = {
    "Modelo operativo": criticos["rank_operativo"].mean(),
    "Reglas heurísticas": criticos["rank_reglas"].mean(),
    "Risk Score ML": criticos["rank_risk"].mean()
}

# Plot
fig4 = px.bar(
    x=list(metricas_ml.keys()),
    y=list(metricas_ml.values()),
    title="Posición media de expedientes críticos",
    labels={
        "x": "Modelo",
        "y": "Posición media (menor = mejor)"
    }
)

st.plotly_chart(fig4, use_container_width=True)
st.info(
    "El modelo ML reduce la posición media de los casos críticos, "
    "mejorando la capacidad de priorización operativa."
)

# FOOTER

st.caption("Sistema IPS - Modelo de asignación inteligente con optimización de carga y priorización operativa")


