# Kenya Maize Yield Predictor (Streamlit App)

Interactive demo wrapping the Gradient Boosting model from
[Kenya Maize Yield Prediction](../Kenya_Maize_Yield_Prediction.ipynb) in a usable interface.
Enter rainfall, NDVI, and terrain values, get a maize yield prediction for
Uasin Gishu County, Kenya.

## Files

- `app.py` — the Streamlit app
- `train_model.py` — reconstructs the exact training dataset from the notebook's output
  and retrains the model (reproduces R² = 0.6875, RMSE = 0.224, matching the original notebook)
- `maize_yield_model.pkl` — trained Gradient Boosting model
- `model_metadata.pkl` — feature names, ranges, and performance metrics
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Deploy for free (so you can link it live on your portfolio)

**Streamlit Community Cloud** (easiest, free):
1. Push this folder to a GitHub repo (or a subfolder of your existing
   `crop-yield-prediction` repo).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app," point it at your repo, branch, and `app.py` path.
4. Deploy. You'll get a public URL like
   `https://your-app-name.streamlit.app` you can link from your portfolio.

## Retraining

If you regenerate the dataset from Earth Engine (more years, different county),
edit `train_model.py` with the new data and rerun it to produce a fresh
`maize_yield_model.pkl`.
