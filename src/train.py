import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_data, clean_data, add_features

# carrega e processa os dados
print("📥 Carregando e processando dados...")
inicio = time.time()
# carga de arquivo XLSX demora muito (pelos testes, em torno de 1 minuto), 
# então vamos usar CSV para agilizar o processo
# df = load_data("dataset/Focos-AmazoniaLegal2020-2025-Final.xlsx")
df = pd.read_csv("dataset/Focos-AmazoniaLegal2020-2025-Final.csv", encoding="utf-8", sep=";", low_memory=False)
print(f"✅ Dados carregados com sucesso!\r\nTotal de registros: {len(df)}\r\nTempo: {time.time() - inicio:.2f} segundos")
print("📊 Processando dados...")
inicio = time.time()
df = clean_data(df)
print(f"✅ Dados limpos!\r\nTempo: {time.time() - inicio:.2f} segundos")
print("🔄 Adicionando características...")
inicio = time.time()
df = add_features(df)
print(f"✅ Características adicionadas!\r\nTempo: {time.time() - inicio:.2f} segundos")

# alvo
df = df[df["FRP"] > 0]  

y = np.log1p(df["FRP"])

# features
numeric_features = ["Mes", "DiaSemChuva", "Precipitacao"]
categorical_features = ["Municipio_UF", "Estacao"]

X = df[numeric_features + categorical_features]

# préprocessamento
preprocessador = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# modelos
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingRegressor(random_state=42)

rf_model = Pipeline([
    ("prep", preprocessador),
    ("model", rf)
])

gb_model = Pipeline([
    ("prep", preprocessador),
    ("model", gb)
])

# divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinamento dos modelos
print("🚀 Iniciando treinamento dos modelos...")
inicio = time.time()
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
print(f"🏁 Treinamento concluído (Tempo: {time.time() - inicio:.2f} segundos")

# =========================
# PREDICTIONS (NO LOG SPACE)
# =========================
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

print("\n📊 RESULTADOS (LOG SPACE)")
print("RF MAE:", mean_absolute_error(y_test, rf_pred))
print("RF R²:", r2_score(y_test, rf_pred))

print("GB MAE:", mean_absolute_error(y_test, gb_pred))
print("GB R²:", r2_score(y_test, gb_pred))

# Salva os modelos em disco
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(gb_model, "models/gb_model.pkl")

# salvar municípios
municipios = sorted(df["Municipio_UF"].unique())
joblib.dump(municipios, "models/municipios.pkl")

print("\n✔ Modelos salvos com sucesso!")