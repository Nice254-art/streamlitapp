import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "model_xgb.pkl"  # Change path if your model is elsewhere
st.set_page_config(page_title="ReGenVision 🌱", layout="wide")

# ----------------------------
# Header
# ----------------------------
st.title("🌍 ReGenVision - AI for Regenerative Land Management")
st.markdown("""
Welcome to **ReGenVision**, an AI-powered solution built during the **ReGen Hackathon 2025**.

This dashboard helps analyze soil health and landscape sustainability using data-driven insights.
""")

# ----------------------------
# Debug: check model path
# ----------------------------
st.write("✅ Checking model path...")
st.write("Model path:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Model file not found. Predictions will be disabled until model_xgb.pkl is available.")

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

model = load_model()

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("📊 Input Environmental Parameters")
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=800.0)
soc = st.sidebar.number_input("Soil Organic Carbon (%)", min_value=0.0, max_value=10.0, value=1.5)
slope = st.sidebar.number_input("Slope (degrees)", min_value=0.0, max_value=45.0, value=10.0)

# ----------------------------
# Predict NDVI / Vegetation Index
# ----------------------------
if st.sidebar.button("🔍 Predict Land Health"):
    if model is None:
        st.error("❌ Cannot make prediction: model not loaded.")
    else:
        input_data = pd.DataFrame([[rainfall, soc, slope]], columns=["Rainfall", "SOC", "Slope"])
        prediction = model.predict(input_data)[0]

        st.subheader("🌾 AI Prediction Result")
        st.metric("Predicted NDVI (Vegetation Health)", f"{prediction:.3f}")
        st.info("Higher NDVI means healthier and greener vegetation 🌿")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
👩🏽‍💻 **Team ReGenVision** | Built for ReGen Hackathon 2025  
Leveraging AI, remote sensing, and open environmental data for sustainable land restoration.
""")
