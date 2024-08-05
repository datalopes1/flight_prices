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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

best_params = {
    'learning_rate': 0.09372862218041428,
    'max_depth': 10,
    'subsample': 0.7358527060931693,
    'colsample_bytree': 0.9494020324700931,
    'min_child_weight': 7
    }

model = XGBRegressor(**best_params)

xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

xgb.fit(X_train, y_train)
# %% Gerando previsões
y_pred = xgb.predict(X_test)
# %% Métricas do modelo
def metrics_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared = False)
    r2 = r2_score(y_true, y_pred)

    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R2 Score': r2
    }

metrics = metrics_report(y_test, y_pred)
metrics
# %%
modelo_pred = pd.Series({
    'model': xgb,
    'features': features,
    'metrics': metrics
})

modelo_pred.to_pickle("../../models/modelo_pred.pkl")