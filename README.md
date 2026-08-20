# East Africa Crop Yield Prediction: Machine Learning for Food Security

**Status**: ✅ Complete | **Model Performance**: R² = 0.82 | **RMSE**: 0.25 tons/ha

![Model Results](results/model_results.png)

## Problem Statement

Kenya's Rift Valley (Uasin Gishu County) produces ~40% of the nation's maize, but yield predictions remain manual and unreliable. Smallholder farmers lack data-driven tools to anticipate harvests given rainfall, vegetation health, and terrain conditions, limiting their ability to plan for food security and climate adaptation.

## Solution

Built an **end-to-end machine learning pipeline** that predicts maize yields using freely available satellite data and climate observations. The model identifies which environmental factors most strongly influence crop productivity, enabling early warning systems and data-driven agricultural policy.

### Key Innovation
Combines **multi-source Earth observation data** (CHIRPS, Sentinel-2, SRTM) with **statistical learning** to create a reproducible, interpretable yield prediction model for East Africa.

---

## Dataset & Methodology

### Data Sources

| Dataset | Source | Resolution | Use |
|---------|--------|-----------|-----|
| **CHIRPS** | UC Santa Barbara | 5 km, daily | Seasonal rainfall (Oct-Mar 2015-2023) |
| **Sentinel-2** | ESA/Copernicus | 10 m, 5-day | Vegetation health (NDVI) during growing season |
| **SRTM** | USGS/NASA | 30 m | Elevation, slope |
| **Kenya KNBS** | National Bureau of Statistics | County-level | Maize yields (2015-2023) |

### Features Engineered
