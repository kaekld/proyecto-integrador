import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Optimización Agrícola", layout="wide")
st.title("Optimización Inferencial del Rendimiento Agrícola")

@st.cache_data
def cargar_datos():
    con = duckdb.connect(':memory:')
    con.execute("SELECT setseed(0.42);")
    
    query = """
    CREATE TABLE parcelas AS
    WITH base AS (
        SELECT 
            range AS parcela_id,
            ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
            ROUND(50.0 + random() * 100.0, 2) AS fertilizante_kg,
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            -0.8 + random() * 1.6 AS ruido
        FROM range(1, 101)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(
            5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + 
            CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END + ruido, 
        2) AS rendimiento_ton
    FROM base;
    """
    con.execute(query)
    return con.execute("SELECT * FROM parcelas").df()

df = cargar_datos()

st.sidebar.header("Filtros de Telemetría")
tratamientos = st.sidebar.multiselect(
    "Seleccionar Tratamiento:",
    options=df["tipo_tratamiento"].unique(),
    default=df["tipo_tratamiento"].unique()
)
riego_min, riego_max = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    float(df["volumen_riego"].min()),
    float(df["volumen_riego"].max()),
    (float(df["volumen_riego"].min()), float(df["volumen_riego"].max()))
)

df_filtrado = df[
    (df["tipo_tratamiento"].isin(tratamientos)) &
    (df["volumen_riego"].between(riego_min, riego_max))
]

X = sm.add_constant(df[['volumen_riego', 'fertilizante_kg']])
y = df['rendimiento_ton']
modelo_ols = sm.OLS(y, X).fit()
r2_adj = modelo_ols.rsquared_adj
residuos = modelo_ols.resid

grupo_a = df[df['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_b = df[df['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']
t_stat, p_valor = stats.ttest_ind(grupo_b, grupo_a, alternative='greater')

col1, col2, col3 = st.columns(3)
col1.metric("Rendimiento Promedio (Filtrado)", f"{df_filtrado['rendimiento_ton'].mean():.2f} ton/ha")
col2.metric("R² Ajustado del Modelo OLS", f"{r2_adj * 100:.2f}%")
col3.metric("P-Valor (Prueba t-Student)", f"{p_valor:.4e}", 
            delta="Significativo (p < 0.05)" if p_valor < 0.05 else "No significativo")

st.divider()

fila1_col1, fila1_col2 = st.columns(2)
fila2_col1, fila2_col2 = st.columns(2)

with fila1_col1:
    st.subheader("1. Correlación de Pearson")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    corr = df_filtrado[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

with fila1_col2:
    st.subheader("2. Regresión: Riego vs Rendimiento")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df_filtrado, x='volumen_riego', y='rendimiento_ton', ax=ax2, 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig2)

with fila2_col1:
    st.subheader("3. Comparativa de Tratamientos (A vs B)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_filtrado, x='tipo_tratamiento', y='rendimiento_ton', palette="Set2", ax=ax3)
    ax3.set_xlabel("Tipo de Tratamiento")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig3)

with fila2_col2:
    st.subheader("4. Diagnóstico de Residuos del Modelo")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuos, kde=True, color="purple", ax=ax4)
    ax4.axvline(0, color='black', linestyle='--')
    ax4.set_xlabel("Residuos (Real - Predicho)")
    st.pyplot(fig4)

with st.expander("Ver Resumen Estadístico Completo (OLS y Veredicto)"):
    st.text(modelo_ols.summary())
    if p_valor < 0.05:
        st.success("**Veredicto de Ingeniería:** Se rechaza H₀. El Tratamiento_B incrementa significativamente el rendimiento. Se recomienda escalar la inversión.")
    else:
        st.warning("**Veredicto de Ingeniería:** No se rechaza H₀. No hay evidencia suficiente para adoptar el Tratamiento_B.")
