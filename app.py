import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Kenya Maize Yield Predictor", page_icon="🌽", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("maize_yield_model.pkl")
    metadata = joblib.load("model_metadata.pkl")
    return model, metadata

model, metadata = load_model()

st.title("🌽 Kenya Maize Yield Predictor")
st.caption("Uasin Gishu County, Rift Valley — Gradient Boosting model trained on CHIRPS rainfall, "
           "Sentinel-2 NDVI, and SRTM terrain data (2015–2023).")

with st.expander("About this model"):
    st.markdown(f"""
    - **Model**: {metadata['model_type']}
    - **Test R²**: {metadata['model_r2']:.4f} (explains ~69% of yield variance)
    - **RMSE**: {metadata['model_rmse']:.3f} tons/hectare
    - **MAE**: {metadata['model_mae']:.3f} tons/hectare
    - **Training data**: {metadata['region']}, {min(metadata['training_years'])}–{max(metadata['training_years'])} ({len(metadata['training_years'])} seasons)
    - **Note**: This is a research prototype trained on only 9 years of county-level data.
      Predictions should be treated as directional estimates, not precise forecasts.
    """)

st.subheader("Enter growing season conditions")

ranges = metadata["feature_ranges"]

col1, col2 = st.columns(2)

with col1:
    rainfall_total = st.slider(
        "Total seasonal rainfall (mm, Oct–Mar)",
        min_value=300, max_value=1300,
        value=700, step=10,
        help=f"Historical range in training data: {ranges['rainfall_total'][0]:.0f}–{ranges['rainfall_total'][1]:.0f} mm"
    )
    rainfall_peak = st.slider(
        "Peak monthly rainfall (mm)",
        min_value=50, max_value=280,
        value=170, step=5,
        help=f"Historical range: {ranges['rainfall_peak'][0]:.0f}–{ranges['rainfall_peak'][1]:.0f} mm. "
             "This is the single strongest predictor (~58% importance)."
    )
    rainfall_cv = st.slider(
        "Rainfall variability (coefficient of variation)",
        min_value=0.2, max_value=0.8,
        value=0.5, step=0.01,
        help="Higher values mean more erratic month-to-month rainfall."
    )

with col2:
    ndvi_mean = st.slider(
        "Mean NDVI, peak growing season (Dec–Feb)",
        min_value=0.3, max_value=0.7,
        value=0.5, step=0.01,
        help=f"Historical range: {ranges['ndvi_mean'][0]:.2f}–{ranges['ndvi_mean'][1]:.2f}. "
             "Higher = healthier vegetation canopy."
    )
    elevation = st.number_input(
        "Elevation (m)",
        value=round(ranges["elevation"]), step=10,
        help="Uasin Gishu County average is ~1,707 m. Adjust only if modeling a different sub-region."
    )
    slope = st.number_input(
        "Slope (degrees)",
        value=round(ranges["slope"], 1), step=0.1,
        help="Uasin Gishu County average is ~5.8°."
    )

if st.button("Predict Yield", type="primary", use_container_width=True):
    X_input = pd.DataFrame([{
        "rainfall_total": rainfall_total,
        "rainfall_cv": rainfall_cv,
        "rainfall_peak": rainfall_peak,
        "ndvi_mean": ndvi_mean,
        "elevation": elevation,
        "slope": slope,
    }])

    prediction = model.predict(X_input)[0]

    st.metric("Predicted Maize Yield", f"{prediction:.2f} tons/hectare")

    # Contextualize against historical range
    if prediction < 3.0:
        st.warning("This falls in the lower range of historical yields (2.8–3.8 tons/ha). "
                   "Conditions suggest below-average season, consider drought mitigation planning.")
    elif prediction > 3.5:
        st.success("This falls in the upper range of historical yields (2.8–3.8 tons/ha). "
                   "Conditions suggest a strong season.")
    else:
        st.info("This falls within the typical historical range (2.8–3.8 tons/ha) for this county.")

st.divider()
st.caption("Built by Jecinta Wanjiru · [Portfolio](https://wanjiru-1.github.io/) · "
           "[Full analysis on Medium](#) · Model: Gradient Boosting Regressor trained in scikit-learn.")
