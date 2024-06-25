# %%
# Importação das bibliotecas
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn import pipeline

from feature_engine import encoding
# %%
#Carregamento dos dados
df = pd.read_csv("../data/processed/train.csv")
df.head()
# %%
# Definindo as features
X = [
 'airline',
 'source_city',
 'departure_time',
 'stops',
 'arrival_time',
 'destination_city',
 'class',
 'duration',
 'days_left'
 ]

y = 'price'

# Dividindo as variáveis e numéricas
cat_features = ['airline',
 'source_city',
 'departure_time',
 'stops',
 'arrival_time',
 'destination_city',
 'class']

num_features = ['duration', 'days_left', 'price']

# Dividindo os conjuntos de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[X], 
                                                                    df[y], 
                                                                    test_size=0.20, 
                                                                    random_state=42)

print(f"Preço média da passagem em treino {y_train.mean()}")
print(f"Preço média da passagem em teste {y_test.mean()}")
# %%
# O pipeline
# One Hot Encoding
onehot = encoding.OneHotEncoder(variables=cat_features)

# Modelo
model = XGBRegressor(random_state=42)

# Pipeline
xgb_pipeline = pipeline.Pipeline([
    ('ohe', onehot),
    ('model', model)
])

xgb_pipeline.fit(X_train, y_train)
# %%
# Predições
y_train_predict = xgb_pipeline.predict(X_train)
y_test_predict = xgb_pipeline.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_predict)
r2_train = r2_score(y_train, y_train_predict)

mse_test = mean_squared_error(y_test, y_test_predict)
r2_test = r2_score(y_test, y_test_predict)
# %%
# Métricas
print("Métricas de Treino")
print("=" * 40)
print(f"Mean Squared Error: {mse_train}")
print(f"R2 Score: {r2_train}")
print("\nMétricas de Teste")
print("=" * 40)
print(f"Mean Squared Error: {mse_test}")
print(f"R2 Score: {r2_test}")
# %%
# Gerando o binário do modelo
model_xgb = pd.Series({
    "model": xgb_pipeline,
    "features": X,
    "r2_score": r2_test
})
model_xgb.to_pickle("xgb_model.pkl")