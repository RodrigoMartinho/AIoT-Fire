import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Comparativo de Risco de Fogo", page_icon="🔥", layout="wide")

#carrega os modelos e a lista de municípios
@st.cache_resource
def load_models():
    rf_model = joblib.load("models/rf_model.pkl")
    gb_model = joblib.load("models/gb_model.pkl")
    municipios = joblib.load("models/municipios.pkl")

    return rf_model, gb_model, municipios

@st.cache_data 
def load_data():
    return pd.read_csv("models/focos_por_mes.csv")

rf_model, gb_model, municipios = load_models()
grafico_df = load_data()

# Dicionário para traduzir o nome do mês para o número que o modelo espera
MAPA_MESES = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4,
    "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8,
    "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
}

#Interface do usuário
st.title("🔥 Sistema de Previsão de Fogo - Amazônia Legal")
st.markdown("Insira os dados ambientais abaixo para comparar as previsões dos modelos **Random Forest** e **Gradient Boosting**.")

if "mes_selecionado" not in st.session_state:
    st.session_state.mes_selecionado = "Agosto"

def set_mes(mes_nome):
    st.session_state.mes_selecionado = mes_nome

with st.sidebar:
    st.header("📥 Dados de Entrada")
    municipio = st.selectbox("Município", municipios, index=None, placeholder="Selecione um município...")
    
    st.markdown("**Mês**")
    meses_layout = [
        ["Janeiro", "Fevereiro", "Março"],
        ["Abril", "Maio", "Junho"],
        ["Julho", "Agosto", "Setembro"],
        ["Outubro", "Novembro", "Dezembro"]
    ]
    
    for linha in meses_layout:
        cols = st.columns(3)
        for i, mes_nome in enumerate(linha):
            with cols[i]:
                st.button(
                    mes_nome[:3], # Pega as 3 primeiras letras (Jan, Fev, Mar...)
                    key=f"btn_{mes_nome}",
                    type="primary" if st.session_state.mes_selecionado == mes_nome else "secondary",
                    use_container_width=True,
                    on_click=set_mes,
                    args=(mes_nome,)
                )

    mes = MAPA_MESES[st.session_state.mes_selecionado]
    dias_sem_chuva = st.slider("Dias sem chuva", 0, 90, 15)
    precipitacao = st.slider("Precipitação (mm)", 0.0, 150.0, 0.0)
    comparar = st.button("Comparar Modelos", type="primary", use_container_width=True)

col_hist, col_comp = st.columns([2, 3])

with col_hist:
    st.subheader("📈 Histórico de focos por mês")

    if municipio:
        # Filtra os dados
        dados_municipio = grafico_df[grafico_df["Municipio_UF"] == municipio]

        # Para o gráfico nativo do Streamlit ficar perfeito, 
        # transformamos a coluna 'Mes' no índice do dataframe
        dados_grafico = dados_municipio.set_index("Mes")["QuantidadeFocos"]

        if not dados_grafico.empty:
            # st.bar_chart (barras) ou st.line_chart (linhas) geram gráficos interativos na hora!
            st.line_chart(dados_grafico, use_container_width=True)
        else:
            st.info("Sem histórico de focos de incêndio para este município.")
    else:
        st.info("👈 Selecione um município para visualizar o histórico.")


def obter_status_risco(frp):
    if frp < 20:
        return "BAIXO 🟢", "success"
    elif frp < 35:
        return "MÉDIO 🟡", "warning"
    else:
        return "ALTO 🔴", "error"

with col_comp:
    st.subheader("📊 Resultado Comparativo")
    
    if comparar:
        if not municipio:
            st.warning("⚠️ Por favor, selecione um município na barra lateral antes de comparar.")
        else:
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
    else:
        st.info("👈 Preencha os dados e clique em **Comparar Modelos**.")