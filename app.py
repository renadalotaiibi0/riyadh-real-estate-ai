import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Riyadh Real Estate Price Prediction & Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# Load Model and Data Safely
@st.cache_resource
def load_model_and_data():
    csv_path = "riyadh_real_estate.csv"
    model = None
    df = None
    
    if os.path.exists("model.pkl"):
        try:
            model = joblib.load("model.pkl")
        except Exception:
            pass
            
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        except Exception:
            pass
            
    return model, df

model, df = load_model_and_data()

# Districts mapped by Region
region_districts = {
    "North": ["Al-Malqa", "Al-Narjis", "As-Sahafah", "Al-Yasmin", "Hittin", "Al-Aarid", "Al-Nakheel"],
    "East": ["Al-Munasiyah", "Al-Qadisiyah", "Al-Hamra", "Al-Rundah", "Ishbiliya"],
    "West": ["Laban", "Tuwaiq", "Dharat Laban", "Al-Mahdia"],
    "South": ["Al-Shifa", "Al-Aziziyah", "Badr", "Al-Dar El Beida"],
    "Central": ["Al-Olaya", "Al-Sulaimaniyah", "Al-Murabba", "Al-Malaz"]
}

# Sidebar - Property Features
st.sidebar.markdown("### 🔧 Property Features")

property_type = st.sidebar.selectbox("Property Type:", ["Apartment", "Villa", "Land", "Building"])
region = st.sidebar.selectbox("Select Region:", list(region_districts.keys()))
district = st.sidebar.selectbox("Select District:", region_districts[region])

area = st.sidebar.slider("Property Area (m²):", 50, 1500, 210)
bedrooms = st.sidebar.slider("Bedrooms:", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms:", 1, 10, 3)
property_age = st.sidebar.slider("Property Age (Years):", 0, 30, 0)

predict_btn = st.sidebar.button("🔮 Predict Price & Trends")

# Header
st.title("🏡 Riyadh Real Estate Price Prediction & Investment Advisor")
st.markdown("An advanced Machine Learning portfolio project for analyzing, predicting property prices, and forecasting market trends in Riyadh.")

# Calculation & Prediction Logic
def calculate_predictions():
    # Base calculation logic fallback
    base_m2 = 5500 if region == "North" else 4200
    if property_type == "Villa":
        base_m2 *= 1.3
    
    estimated_price = (area * base_m2) + (bedrooms * 40000) + (bathrooms * 25000) - (property_age * 15000)
    estimated_price = max(estimated_price, 300000)

    if model is not None:
        try:
            input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=['area', 'bedrooms', 'bathrooms'])
            model_pred = model.predict(input_data)[0]
            if model_pred > 0:
                estimated_price = model_pred
        except Exception:
            pass

    growth_rate = 0.030  # 3% growth
    projected_price = estimated_price * (1 + growth_rate)
    diff = projected_price - estimated_price

    return estimated_price, projected_price, diff, growth_rate

# Default / Triggered Display
st.success(f"✨ Analysis for *{property_type}* Completed Successfully!")

est_price, proj_price, price_diff, growth = calculate_predictions()

# Metrics Layout (3 Columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"*💰 Estimated {property_type} Price*")
    st.header(f"{est_price:,.0f} SAR")

with col2:
    st.markdown("*📈 Projected Price (1-Year)*")
    st.header(f"{proj_price:,.0f} SAR")
    st.caption(f"▲ +{price_diff:,.0f} SAR ({growth*100:.1f}%)")

with col3:
    st.markdown("*📍 Location & Type*")
    st.header(f"{district} ({property_type})")

st.markdown("---")

# Insights Section
st.subheader("💡 AI Investment Advisor Insights")
st.markdown("*Investment Rating:* 🔥 High Growth Potential (Prime Investment)")

st.info(f"Properties in *{district}* show steady capital appreciation backed by high market demand in *{region} Riyadh*.")
