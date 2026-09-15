# Kenya Maize Yield Prediction — Geospatial AI

Exploratory geospatial machine-learning study of maize yield variation in **Uasin Gishu County, Kenya**, combining Earth observation, climate data and historical county-level yield observations.

## What changed in the revised workflow

The notebook now uses:

- **Google Earth Engine + geemap** for reproducible geospatial analysis and interactive mapping.
- **geoBoundaries ADM1** to define Uasin Gishu County and contextualize it within Kenya.
- **CHIRPS Daily** rainfall for the October–March growing season, summarized as seasonal total, peak monthly rainfall and rainfall variability.
- **Sentinel-2 Surface Reflectance + cloud probability masking** for December–February NDVI.
- **SRTM** elevation and slope for terrain context. These are excluded from the annual predictive feature set because they do not vary across years at county level.
- **Random Forest and Gradient Boosting** models evaluated with **Leave-One-Out Cross-Validation (LOOCV)** because only nine annual yield observations are available.
- Out-of-sample **R², RMSE and MAE** for model evaluation.

## Important result

The revised analysis is intentionally conservative. With only **9 annual county-level observations (2015–2023)**, the current model should be treated as an **exploratory research prototype**, not an operational yield-forecasting system.

The revised notebook reports the actual LOOCV performance generated from the current dataset rather than the previous 80/20 holdout result. A negative out-of-sample R² indicates that the current model does not yet outperform a simple mean-yield baseline reliably.

Feature importance is also interpreted as descriptive model behavior, **not causal evidence**.

## Repository contents

- `notebooks/Kenya_Maize_Yield_Prediction.ipynb` — revised end-to-end analysis.
- `app.py` — optional Streamlit interface from the earlier prototype.
- `train_model.py` — earlier training/reconstruction script; update it before using it as a production pipeline because the notebook methodology has changed.
- `requirements.txt` — project dependencies.

## Scope

The current target is **county-season maize yield (tons/ha)**. It does not provide field-level or 10 m yield prediction. Scaling to finer spatial units would require geographically distributed yield observations and independent spatial validation.

## Future work

Potential extensions include longer yield time series, ward/field-level observations, soil moisture and texture, temperature, evapotranspiration, phenology, planting dates, crop varieties, and spatial validation.

## Reproducibility

The notebook is designed for Google Colab and requires access to a Google Earth Engine project. Before publication, verify the historical yield values against the exact KNBS / Ministry of Agriculture source used for the study.
