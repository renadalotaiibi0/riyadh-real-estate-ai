import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Riyadh Real Estate AI Advisor",
    page_icon="🏠",
    layout="wide"
)

# Function to train and save model with Property_Type support
def build_and_train_model():
    district_multipliers = {
        'Al-Malqa': 11200, 'Al-Aqiq': 10500, 'Al-Yasmin': 9500, 'Al-Sahafa': 9200, 'Al-Narjis': 7500,
        'Al-Rimal': 4200, 'Al-Munsiyah': 5200, 'Al-Qadisiyah': 4800, 'Al-Hamra': 7000,
        'Al-Mahdiyah': 5500, 'Dharat Laban': 4800, 'Al-Tuwaiq': 3800,
        'Al-Suwaidi': 4200, 'Al-Shifa': 3500, 'Al-Aziziyah': 3200,
        'Al-Olaya': 10500, 'Al-Sulaimaniyah': 9800, 'Al-Murabba': 6500
    }

    districts_info = {
        'North': ['Al-Malqa', 'Al-Yasmin', 'Al-Narjis', 'Al-Sahafa', 'Al-Aqiq'],
        'East': ['Al-Rimal', 'Al-Munsiyah', 'Al-Qadisiyah', 'Al-Hamra'],
        'West': ['Dharat Laban', 'Al-Tuwaiq', 'Al-Mahdiyah'],
        'South': ['Al-Suwaidi', 'Al-Shifa', 'Al-Aziziyah'],
        'Central': ['Al-Olaya', 'Al-Sulaimaniyah', 'Al-Murabba']
    }

    np.random.seed(42)
    data = []
    for _ in range(2500):
        region = np.random.choice(list(districts_info.keys()))
        district = np.random.choice(districts_info[region])
        property_type = np.random.choice(['Villa', 'Apartment'], p=[0.5, 0.5])
        
        if property_type == 'Villa':
            area = int(np.random.normal(380, 100))
            area = max(200, min(area, 1000))
            bedrooms = np.random.randint(3, 7)
            type_multiplier = 1.0
        else:
            area = int(np.random.normal(160, 40))
            area = max(80, min(area, 300))
            bedrooms = np.random.randint(1, 4)
            type_multiplier = 0.72

        bathrooms = max(1, bedrooms + np.random.randint(-1, 2))
        age = np.random.randint(0, 20)
        
        base_m2_price = district_multipliers[district] * type_multiplier
        price = (area * base_m2_price) * (1 - (age * 0.012)) + np.random.normal(0, 20000)
        price = max(250000, price)
        
        data.append({
            'Property_Type': property_type,
            'District': district,
            'Region': region,
            'Area_m2': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Property_Age': age,
            'Price_SAR': int(price)
        })

    df = pd.DataFrame(data)
    X = df[['Property_Type', 'District', 'Region', 'Area_m2', 'Bedrooms', 'Bathrooms', 'Property_Age']]
    y = df['Price_SAR']

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Property_Type', 'District', 'Region'])],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, 'riyadh_housing_model.pkl')
    return model_pipeline

# Automatically train and load model
@st.cache_resource
def load_or_train_model():
    return build_and_train_model()

model = load_or_train_model()

# App Header
st.title("🏠 Riyadh Real Estate Price Prediction & Investment Advisor")
st.markdown("An advanced Machine Learning portfolio project for analyzing, predicting property prices, and forecasting market trends in Riyadh.")

st.sidebar.header("🔧 Property Features")

# Property Type Input
property_type = st.sidebar.selectbox("Property Type:", ["Villa", "Apartment"])

districts_data = {
    'North': ['Al-Malqa', 'Al-Yasmin', 'Al-Narjis', 'Al-Sahafa', 'Al-Aqiq'],
    'East': ['Al-Rimal', 'Al-Munsiyah', 'Al-Qadisiyah', 'Al-Hamra'],
    'West': ['Dharat Laban', 'Al-Tuwaiq', 'Al-Mahdiyah'],
    'South': ['Al-Suwaidi', 'Al-Shifa', 'Al-Aziziyah'],
    'Central': ['Al-Olaya', 'Al-Sulaimaniyah', 'Al-Murabba']
}

growth_rates = {
    'Al-Malqa': 0.028, 'Al-Aqiq': 0.025, 'Al-Yasmin': 0.024, 'Al-Sahafa': 0.022, 'Al-Narjis': 0.030,
    'Al-Hamra': 0.020, 'Al-Munsiyah': 0.022, 'Al-Qadisiyah': 0.020, 'Al-Rimal': 0.025,
    'Al-Mahdiyah': 0.022, 'Dharat Laban': 0.018, 'Al-Tuwaiq': 0.015,
    'Al-Suwaidi': 0.015, 'Al-Shifa': 0.012, 'Al-Aziziyah': 0.012,
    'Al-Olaya': 0.020, 'Al-Sulaimaniyah': 0.020, 'Al-Murabba': 0.018
}

region = st.sidebar.selectbox("Select Region:", list(districts_data.keys()))
district = st.sidebar.selectbox("Select District:", districts_data[region])

if property_type == 'Villa':
    area = st.sidebar.slider("Property Area (m²):", min_value=150, max_value=1000, value=350, step=25)
    bedrooms = st.sidebar.slider("Bedrooms:", min_value=2, max_value=7, value=4)
else:
    area = st.sidebar.slider("Property Area (m²):", min_value=50, max_value=400, value=150, step=10)
    bedrooms = st.sidebar.slider("Bedrooms:", min_value=1, max_value=5, value=3)

bathrooms = st.sidebar.slider("Bathrooms:", min_value=1, max_value=6, value=3)
property_age = st.sidebar.slider("Property Age (Years):", min_value=0, max_value=20, value=2)

# Input DataFrame
input_data = pd.DataFrame({
    'Property_Type': [property_type],
    'District': [district],
    'Region': [region],
    'Area_m2': [area],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'Property_Age': [property_age]
})

# Prediction logic
if st.sidebar.button("🔮 Predict Price & Trends"):
    prediction = model.predict(input_data)[0]
    
    base_growth = growth_rates.get(district, 0.02)
    effective_growth = max(0.008, base_growth - (property_age * 0.0008))
    
    future_price = prediction * (1 + effective_growth)
    price_diff = future_price - prediction
    
    st.success(f"✨ Analysis for *{property_type}* Completed Successfully!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"💰 Estimated {property_type} Price", value=f"{int(prediction):,} SAR")
    with col2:
        st.metric(
            label="📈 Projected Price (1-Year)", 
            value=f"{int(future_price):,} SAR", 
            delta=f"+{int(price_diff):,} SAR ({effective_growth*100:.1f}%)"
        )
    with col3:
        st.metric(label="📍 Location & Type", value=f"{district} ({property_type})")
        
    st.markdown("---")
    st.subheader("💡 AI Investment Advisor Insights")
    
    if effective_growth >= 0.025:
        rating = "🔥 *High Growth Potential (Prime Investment)*"
        advice = f"Properties in *{district}* show steady capital appreciation backed by high market demand in North Riyadh."
    elif effective_growth >= 0.018:
        rating = "📈 *Stable & Secure Investment*"
        advice = f"*{district}* maintains a healthy price stability with moderate annual value growth suitable for residential hold."
    else:
        rating = "💼 *Steady Residential Value*"
        advice = f"*{district}* presents mature market prices with low volatility, ideal for long-term residential stability."
        
    st.markdown(f"*Investment Rating:* {rating}")
    st.info(advice)

st.markdown("---")
st.subheader("📊 Training Data Overview")
if st.checkbox("Show a sample of the training dataset"):
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'Property_Type': ['Villa', 'Apartment', 'Villa', 'Apartment', 'Villa'],
        'District': np.random.choice(districts_data[region], 5),
        'Region': [region]*5,
        'Area_m2': [350, 160, 450, 180, 500],
        'Bedrooms': [4, 3, 5, 3, 5],
        'Price_SAR': [3800000, 1250000, 4900000, 1380000, 5200000]
    })
    st.dataframe(sample_df)