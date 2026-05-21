import pandas as pd

def load_data(path):
    return pd.read_excel(path)


def clean_data(df):
    df = df.dropna(subset=["FRP"])

    df["Mes"] = df["Mes"].astype(int)
    df["DiaSemChuva"] = df["DiaSemChuva"].astype(float)
    df["Precipitacao"] = df["Precipitacao"].astype(float)

    return df


def converter_nome_para_sigla(df, coluna_estado="Estado"):
    """
    Mapeia os nomes cheios dos estados da Amazônia Legal para suas respectivas siglas.
    """
    mapeamento_uf = {
        "ACRE": "AC",
        "AMAPÁ": "AP",
        "AMAZONAS": "AM",
        "MARANHÃO": "MA",
        "MATO GROSSO": "MT",
        "PARÁ": "PA",
        "RONDÔNIA": "RO",
        "RORAIMA": "RR",
        "TOCANTINS": "TO"
    }
    
    df[coluna_estado] = df[coluna_estado].astype(str).str.strip()
    df[coluna_estado] = df[coluna_estado].map(mapeamento_uf).fillna(df[coluna_estado])
    
    return df

def add_features(df):
    def estacao(mes):
        if mes in [11, 12, 1, 2, 3, 4, 5]:
            return "chuvosa"
        return "seca"

    df["Estacao"] = df["Mes"].apply(estacao)
    df = converter_nome_para_sigla(df, coluna_estado="Estado")
    df["Municipio_UF"] = df["Municipio"].astype(str) + " - " + df["Estado"].astype(str)    

    return df