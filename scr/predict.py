# %%
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, r2_score
# %%
model = joblib.load("xgb_model.pkl")
# %%
test_data = pd.read_csv("../data/processed/test.csv")
# %%
X_test = test_data[model['features']]
y_test = test_data['price']
# %%
y_test_pred = model['model'].predict(X_test)
y_test_pred
# %%
test_data['pred_price'] = y_test_pred
test_data.to_csv("../data/processed/model_predictions.csv")
# %%
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Métricas no Conjunto de Teste")
print("=" * 40)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
# %%
