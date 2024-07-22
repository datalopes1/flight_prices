#%% Importação das bibliotecas
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
# %% Carregamento do modelo e dos dados
model = pd.read_pickle("../../models/modelo_pred.pkl")
df = pd.read_csv("../../data/raw/Clean_Dataset.csv")
# %% 
X = df[model['features']]
y = df.price
# %%
y_pred = model['model'].predict(X)
# %%
df['pred_price'] = y_pred
df.to_excel("../../data/processed/model_predictions.xlsx", index = False)
# %%
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Métricas no Conjunto de Teste")
print("=" * 40)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")