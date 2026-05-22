import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuração da página para um layout mais amplo
st.set_page_config(page_title="Comparativo de Risco de Fogo", page_icon="🔥", layout="wide")

#carrega os modelos e a lista de municípios
@st.cache_resource
def load_models():
    rf_model = joblib.load("models/rf_model.pkl")
    gb_model = joblib.load("models/gb_model.pkl")
    municipios = joblib.load("models/municipios.pkl")

    return rf_model, gb_model, municipios

rf_model, gb_model, municipios = load_models()
grafico_df = pd.read_csv("models/focos_por_mes.csv")

# Dicionário para traduzir o nome do mês para o número que o modelo espera
MAPA_MESES = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4,
    "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8,
    "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
}

#Interface do usuário
st.title("🔥 Sistema de Previsão de Fogo - Amazônia Legal")
st.markdown("Insira os dados ambientais abaixo para comparar as previsões dos modelos **Random Forest** e **Gradient Boosting**.")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📥 Dados de Entrada")
    municipio = st.selectbox("Município", municipios)
    mes_nome = st.selectbox("Mês", list(MAPA_MESES.keys()), index=7)
    mes = MAPA_MESES[mes_nome]
    dias_sem_chuva = st.slider("Dias sem chuva", 0, 90, 15)
    precipitacao = st.slider("Precipitação (mm)", 0.0, 150.0, 0.0)
    comparar = st.button("Comparar Modelos", type="primary", use_container_width=True)

with right_col:
    st.subheader("📈 Histórico de focos por mês")
    dados_municipio = grafico_df[
        grafico_df["Municipio_UF"] == municipio
    ]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(
        dados_municipio["Mes"],
        dados_municipio["QuantidadeFocos"],
        marker="o"
    )
    ax.set_xlabel("Mês")
    ax.set_ylabel("Quantidade de focos")
    ax.set_xticks(range(1, 13))
    st.pyplot(fig)


def obter_status_risco(frp):
    if frp < 20:
        return "BAIXO 🟢", "success"
    elif frp < 35:
        return "MÉDIO 🟡", "warning"
    else:
        return "ALTO 🔴", "error"

if comparar:
    features_order = ["Mes", "DiaSemChuva", "Precipitacao", "Municipio_UF", "Estacao"]
    
    # 'mes' já é inteiro (1 a 12)
    input_df = pd.DataFrame([{
        "Municipio_UF": municipio,
        "Estacao": "seca" if mes in [6, 7, 8, 9, 10] else "chuvosa",
        "Mes": mes,
        "DiaSemChuva": dias_sem_chuva,
        "Precipitacao": precipitacao
    }])
    
    input_df = input_df[features_order]

    # Executando as Previsões
    rf_frp_log = rf_model.predict(input_df)[0]
    rf_frp = np.expm1(rf_frp_log)
    rf_risco, rf_tipo = obter_status_risco(rf_frp)

    gb_frp_log = gb_model.predict(input_df)[0]
    gb_frp = np.expm1(gb_frp_log)
    gb_risco, gb_tipo = obter_status_risco(gb_frp)

    st.markdown("---")
    st.subheader("📊 Resultado Comparativo")

    col_rf, col_gb = st.columns(2)

    with col_rf:
        st.markdown("### 🌲 Random Forest")
        st.metric(label="FRP Previsto", value=f"{rf_frp:.2f} MW")
        if rf_tipo == "success": st.success(f"Risco: {rf_risco}")
        elif rf_tipo == "warning": st.warning(f"Risco: {rf_risco}")
        else: st.error(f"Risco: {rf_risco}")

    with col_gb:
        st.markdown("### ⚡ Gradient Boosting")
        st.metric(label="FRP Previsto", value=f"{gb_frp:.2f} MW")
        if gb_tipo == "success": st.success(f"Risco: {gb_risco}")
        elif gb_tipo == "warning": st.warning(f"Risco: {gb_risco}")
        else: st.error(f"Risco: {gb_risco}")