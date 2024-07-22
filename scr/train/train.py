#%% Bibliotecas
# Manipulação dos dados
import pandas as pd
import numpy as np

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Machine Learning
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
# %%
df = pd.read_csv("../../data/raw/Clean_Dataset.csv")
df
# %% Seleção das features
features = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left']
target = 'price'

X = df[features]
y = df[target]
# %% Classificação das features
cat_features = X.select_dtypes(include = 'object').columns.to_list()
num_features = X.select_dtypes(include = 'number').columns.to_list()
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %% Pipeline do modelo
num_transformer = Pipeline([
    ('imput', MeanMedianImputer(imputation_method='median')),
])

cat_transformer = Pipeline([
    ('imput', CategoricalImputer(imputation_method='frequent')),
    ('ohe', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

model = XGBRegressor()

xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

xgb.fit(X_train, y_train)
# %% Gerando previsões
y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)
# %% Métricas do modelo
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Métricas do Modelo")
print("=" * 50)
print(f'Mean Squared Error em Treino: {mse_train}')
print(f'Mean Squared Error em Teste: {mse_test}')
print("-" * 50)
print(f'R2 Score em Treino: {r2_train}')
print(f'R2 Score em Teste: {r2_test}')

fig, ax = plt.subplots(figsize = (12, 6))

sns.scatterplot(x = y_test, y = y_pred_test)
ax.set_title('Real x Predito', loc = 'left', fontsize = 16, pad = 12)
ax.set_xlabel('Real', fontsize = 8)
ax.set_ylabel('Predito', fontsize = 8)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'red')
plt.show()
# %%
modelo_pred = pd.Series({
    'model': xgb,
    'features': features,
    'r2_score': r2_test
})

modelo_pred.to_pickle("../../models/modelo_pred.pkl")