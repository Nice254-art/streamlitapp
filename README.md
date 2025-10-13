# 🌍 ReGenVision - AI for Regenerative Land Management

**Live App:** [https://soilhealt.streamlit.app/](https://soilhealt.streamlit.app/)

ReGenVision is an AI-powered dashboard that helps assess **soil health** and **landscape sustainability** using machine learning and open environmental data.  
Built for the **ReGen Hackathon 2025**, this project aims to empower sustainable land management and restoration decisions across Africa 🌿.

---

## ✨ Features

- 🔍 Predict **land vegetation health (NDVI)** based on soil and environmental parameters  
- 📊 Interactive dashboard built with **Streamlit**  
- 🧠 Machine Learning model trained with **XGBoost / scikit-learn**  
- ☁️ Deployed online via **Streamlit Cloud**  
- 💾 Supports loading a local trained model (`model_xgb.pkl`)

---

## 🧩 How It Works

Users can input:
- **Rainfall (mm)**
- **Soil Organic Carbon (%)**
- **Slope (degrees)**

The app then uses a pre-trained model to predict **NDVI (Normalized Difference Vegetation Index)** — a key indicator of land health and vegetation vitality.

Higher NDVI → Healthier vegetation 🌱  
Lower NDVI → Degraded land or poor vegetation 🏜️

---

## 🖥️ Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | Streamlit |
| Backend | Python |
| ML Model | XGBoost / Scikit-learn |
| Data | Environmental and soil datasets |
| Deployment | Streamlit Cloud |

---

## ⚙️ Installation (Run Locally)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/soilhealt.git
   cd soilhealt
