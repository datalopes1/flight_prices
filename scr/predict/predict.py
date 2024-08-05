#%% Importação das bibliotecas
import pandas as pd
# %% Carregamento do modelo e dos dados
model = pd.read_pickle("../../models/modelo_pred.pkl")
df = pd.read_csv("../../data/raw/Clean_Dataset.csv")
# %% 
X = df[model['features']]
# %%
y_pred = model['model'].predict(X)
# %%
df['pred_price'] = y_pred
df.to_excel("../../data/processed/model_predictions.xlsx", index = False)