"""
Reconstructs the training dataset from the original notebook's output values
and retrains the Gradient Boosting model, saving it for use by the Streamlit app.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Exact feature values as computed in the original notebook (from CHIRPS, Sentinel-2, SRTM)
data = [
    {"year": 2015, "rainfall_total": 690.267117, "rainfall_cv": 0.439175, "rainfall_peak": 166.516597, "ndvi_mean": 0.600890, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 3.2},
    {"year": 2016, "rainfall_total": 450.861898, "rainfall_cv": 0.621286, "rainfall_peak": 108.045587, "ndvi_mean": 0.433934, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 2.8},
    {"year": 2017, "rainfall_total": 621.560390, "rainfall_cv": 0.723477, "rainfall_peak": 197.953527, "ndvi_mean": 0.464949, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 3.5},
    {"year": 2018, "rainfall_total": 487.754397, "rainfall_cv": 0.496025, "rainfall_peak": 132.519306, "ndvi_mean": 0.489334, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 3.1},
    {"year": 2019, "rainfall_total": 1141.986824, "rainfall_cv": 0.433872, "rainfall_peak": 245.039063, "ndvi_mean": 0.556574, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 2.9},
    {"year": 2020, "rainfall_total": 828.794362, "rainfall_cv": 0.606718, "rainfall_peak": 226.038553, "ndvi_mean": 0.541192, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 3.4},
    {"year": 2021, "rainfall_total": 652.070774, "rainfall_cv": 0.417606, "rainfall_peak": 160.917914, "ndvi_mean": 0.491184, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 3.8},
    {"year": 2022, "rainfall_total": 703.897085, "rainfall_cv": 0.692688, "rainfall_peak": 198.665998, "ndvi_mean": 0.465580, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 3.6},
    {"year": 2023, "rainfall_total": 846.781979, "rainfall_cv": 0.317270, "rainfall_peak": 167.953707, "ndvi_mean": 0.569513, "elevation": 1706.574604, "slope": 5.774056, "yield_tons_per_ha": 3.3},
]

df = pd.DataFrame(data)

feature_cols = ["rainfall_total", "rainfall_cv", "rainfall_peak", "ndvi_mean", "elevation", "slope"]
X = df[feature_cols]
y = df["yield_tons_per_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
r2_rf = r2_score(y_test, rf.predict(X_test))

gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print(f"Random Forest R²: {r2_rf:.4f}")
print(f"Gradient Boosting R²: {r2_gb:.4f}, RMSE: {rmse_gb:.4f}, MAE: {mae_gb:.4f}")

best_model = gb if r2_gb > r2_rf else rf
best_name = "Gradient Boosting" if r2_gb > r2_rf else "Random Forest"

joblib.dump(best_model, "/home/claude/maize_app/maize_yield_model.pkl")

metadata = {
    "feature_names": feature_cols,
    "model_type": best_name,
    "model_r2": float(r2_gb),
    "model_rmse": float(rmse_gb),
    "model_mae": float(mae_gb),
    "region": "Uasin Gishu County, Kenya",
    "training_years": list(range(2015, 2024)),
    "feature_ranges": {
        "rainfall_total": [float(X["rainfall_total"].min()), float(X["rainfall_total"].max())],
        "rainfall_cv": [float(X["rainfall_cv"].min()), float(X["rainfall_cv"].max())],
        "rainfall_peak": [float(X["rainfall_peak"].min()), float(X["rainfall_peak"].max())],
        "ndvi_mean": [float(X["ndvi_mean"].min()), float(X["ndvi_mean"].max())],
        "elevation": float(X["elevation"].mean()),
        "slope": float(X["slope"].mean()),
    },
}
joblib.dump(metadata, "/home/claude/maize_app/model_metadata.pkl")
print("Saved model + metadata.")
