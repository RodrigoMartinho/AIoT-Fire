import streamlit as st
import pandas as pd
import numpy as np
import joblib

# carrega os modelos
rf_model = joblib.load("models/rf_model.pkl")
gb_model = joblib.load("models/gb_model.pkl")
municipios = joblib.load("models/municipios.pkl")

# Interface do usuário
st.title("🔥 Sistema de Previsão de Fogo - Amazônia Legal")

model_choice = st.selectbox(
    "Escolha o modelo",
    ["Random Forest", "Gradient Boosting"]
)

model = rf_model if model_choice == "Random Forest" else gb_model

municipio = st.selectbox("Município", municipios)
mes = st.selectbox("Mês", list(range(1, 13)))
dias_sem_chuva = st.slider("Dias sem chuva", 0, 30, 5)
precipitacao = st.slider("Precipitação (mm)", 0.0, 50.0, 5.0)

# Predição e classificação de risco
if st.button("Prever risco de fogo"):

    input_df = pd.DataFrame([{
        "Municipio": municipio,
        "Estacao": "seca" if mes in [6, 7, 8, 9, 10] else "chuvosa",
        "Mes": mes,
        "DiaSemChuva": dias_sem_chuva,
        "Precipitacao": precipitacao
    }])

    # previsão em log
    frp_log = model.predict(input_df)[0]

    # inversão do log
    frp = np.expm1(frp_log)

    # classificação de risco
    if frp < 20:
        risco = "BAIXO 🟢"
    elif frp < 35:
        risco = "MÉDIO 🟡"
    else:
        risco = "ALTO 🔴"

    st.subheader(f"🔥 FRP previsto: {frp:.2f}")
    st.subheader(f"⚠️ Risco de fogo: {risco}")