"""
Delivery Time Prediction — Interactive Demo
Portfolio showcase app built with Streamlit.

BEFORE RUNNING:
  1. Place your trained model file (e.g. cat_native_optimal.cbm) in the
     `models/` folder and update MODEL_PATH below.
  2. Update FEATURE_CONFIG to match your actual feature names, types,
     and category options (copy these from your training notebook).
  3. If you saved a background sample of X_train for SHAP, point
     SHAP_BACKGROUND_PATH at it. Otherwise SHAP will build one from
     the example inputs (slower on first load, fine for a demo).
"""

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# ------------------------------------------------------------------
# CONFIG — edit this section to match your project
# ------------------------------------------------------------------

MODEL_PATH = "models/catboost_native_final_model.pkl"

FEATURE_CONFIG = {
    "Delivery_person_Age": {"type": "number", "min": 18, "max": 50, "default": 30},
    "Delivery_person_Ratings": {"type": "number", "min": 1.0, "max": 5.0, "default": 4.6, "step": 0.1},
    "Weather_conditions": {"type": "category", "options": ["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"]},
    "Road_traffic_density": {"type": "category", "options": ["Low", "Medium", "High", "Jam"]},
    "Vehicle_condition": {"type": "category", "options": [0, 1, 2]},
    "Type_of_order": {"type": "category", "options": ["Meal", "Drinks", "Buffet", "Snack"]},
    "Type_of_vehicle": {"type": "category", "options": ["motorcycle", "scooter", "electric_scooter"]},
    "multiple_deliveries": {"type": "category", "options": [0, 1, 2, 3]},
    "Festival": {"type": "category", "options": ["No", "Yes"]},
    "City": {"type": "category", "options": ["Metropolitan", "Urban", "Semi-Urban"]},
    "Distance_km": {"type": "number", "min": 0.5, "max": 25.0, "default": 5.0, "step": 0.5},
    "Prep_time_min": {"type": "number", "min": 5, "max": 20, "default": 10},
    "Order_hour": {"type": "number", "min": 0, "max": 23, "default": 13},
}

# Metrics from your evaluation — update with your real numbers
PERFORMANCE = {
    "MAE": 4.71,
    "RMSE": 5.92,
    "R2": 0.598,
    "MAPE": 0.2062,
}

st.set_page_config(page_title="Delivery Time Predictor", layout="wide")

# ------------------------------------------------------------------
# MODEL LOADING (cached so it only loads once per session)
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Convert raw form inputs into the exact feature row your model expects.
    Adjust this to mirror your training-time preprocessing (cyclical hour
    encoding, Festival mapping, etc.)."""
    row = dict(inputs)

    # cyclical hour encoding, mirroring training pipeline
    hour = row.pop("Order_hour")
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # binary mapping, mirroring training pipeline
    row["Festival"] = 1 if row["Festival"] == "Yes" else 0

    return pd.DataFrame([row])


model = load_model()

# ------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------

st.title("🛵 Delivery Time Predictor")
st.caption("A CatBoost model trained to predict food delivery time from order, weather, and traffic features.")

tab_predict, tab_performance, tab_importance, tab_about = st.tabs(
    ["🔮 Live Prediction", "📊 Model Performance", "🧠 Feature Importance", "ℹ️ About"]
)

# ------------------------------------------------------------------
# TAB 1 — Live prediction
# ------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter order details")

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    inputs = {}

    for i, (feat, cfg) in enumerate(FEATURE_CONFIG.items()):
        target_col = cols[i % 3]
        with target_col:
            if cfg["type"] == "category":
                inputs[feat] = st.selectbox(feat, cfg["options"])
            else:
                inputs[feat] = st.number_input(
                    feat,
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["default"],
                    step=cfg.get("step", 1),
                )

    if st.button("Predict delivery time", type="primary"):
        X_input = build_feature_row(inputs)
        pred = model.predict(X_input)[0]

        mae = PERFORMANCE["MAE"]
        low, high = pred - mae, pred + mae

        st.metric("Predicted delivery time", f"{pred:.1f} min", help=f"± {mae:.1f} min typical error (MAE)")
        st.info(f"Likely range: **{low:.1f} – {high:.1f} minutes**")

        # SHAP explanation for this single prediction
        st.subheader("Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[0], feature_names=X_input.columns, show=False
        )
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)

# ------------------------------------------------------------------
# TAB 2 — Model performance
# ------------------------------------------------------------------
with tab_performance:
    st.subheader("Overall test-set performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{PERFORMANCE['MAE']:.2f} min")
    c2.metric("RMSE", f"{PERFORMANCE['RMSE']:.2f} min")
    c3.metric("R²", f"{PERFORMANCE['R2']*100:.1f}%")
    c4.metric("MAPE", f"{PERFORMANCE['MAPE']*100:.1f}%")

    st.markdown("---")
    st.markdown(
        """
        **Why MAE as the primary metric?**
        Delivery time has legitimate long-tail noise (traffic incidents, restaurant
        delays) that RMSE over-penalizes. MAE gives a more representative picture
        of typical accuracy and is easier to communicate to stakeholders
        ("we're off by ~4.7 minutes on average").
        """
    )

    st.info(
        "Drop in your error-by-segment charts here (MAE by weather, traffic, "
        "distance bucket, etc.) — e.g. save them as PNGs during training and "
        "st.image() them, or recompute live if you have error_df available."
    )

# ------------------------------------------------------------------
# TAB 3 — Feature importance
# ------------------------------------------------------------------
with tab_importance:
    st.subheader("What drives delivery time predictions?")
    st.markdown(
        "SHAP summary plot, computed on a sample of the test set. "
        "`hour_sin` / `hour_cos` jointly represent time-of-day (cyclically encoded)."
    )
    st.info(
        "Precompute this on a background sample and save as a static image for "
        "fast app load, or compute live if your dataset is small enough."
    )

# ------------------------------------------------------------------
# TAB 4 — About
# ------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        This app demonstrates an end-to-end ML workflow for predicting food
        delivery time:

        - **Feature engineering**: cyclical encoding for hour-of-day
          (`sin`/`cos`) to correctly represent midnight wraparound; ordinal
          encoding for traffic density; one-hot encoding for nominal categories.
        - **Model selection**: compared Ridge, Lasso, and CatBoost; CatBoost's
          native categorical handling and non-linear splits outperformed linear
          baselines, which showed no meaningful overfitting to regularize away.
        - **Metric choice**: MAE over RMSE, since large errors in this dataset
          reflect real-world noise (traffic incidents, prep delays) rather than
          model failure, and MAE keeps a few outliers from dominating training.
        - **Explainability**: SHAP for both global feature importance and
          per-prediction explanations.

        Built with CatBoost, SHAP, and Streamlit.
        """
    )
