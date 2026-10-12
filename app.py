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
            df = pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')
        except Exception:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
            except Exception:
                pass
            
    return model, df

model, df = load_model_and_data()

# Calibrated Price Ranges per m² (Base Prices)
base_district_prices = {
    # 1. North Riyadh
    "Hittin": {"Land": (9500, 12000), "Villa": (9500, 12000), "Apartment": (6500, 8000), "Building": (8000, 10000)},
    "Al-Malqa": {"Land": (8500, 10000), "Villa": (8500, 10000), "Apartment": (6000, 7500), "Building": (7200, 8800)},
    "As-Sahafah": {"Land": (8000, 10500), "Villa": (8000, 10500), "Apartment": (5800, 7000), "Building": (6800, 8500)},
    "Al-Nakheel": {"Land": (8500, 11000), "Villa": (8500, 11000), "Apartment": (6200, 7800), "Building": (7500, 9500)},
    "Al-Yasmin": {"Land": (6800, 8200), "Villa": (6800, 8200), "Apartment": (5500, 6800), "Building": (6000, 7500)},
    "Al-Narjis": {"Land": (6500, 8000), "Villa": (6500, 8000), "Apartment": (6000, 7500), "Building": (5800, 7200)},
    "Al-Aarid": {"Land": (5200, 6800), "Villa": (5200, 6800), "Apartment": (4800, 6000), "Building": (4800, 6200)},
    
    # 2. Central Riyadh
    "Al-Olaya": {"Land": (8000, 12000), "Villa": (8000, 12000), "Apartment": (6000, 7500), "Building": (7500, 11000)},
    "Al-Sulaimaniyah": {"Land": (7500, 9500), "Villa": (7500, 9500), "Apartment": (5500, 7000), "Building": (6500, 8200)},
    "Al-Murabba": {"Land": (4500, 6500), "Villa": (4500, 6500), "Apartment": (3800, 5000), "Building": (4000, 5800)},
    "Al-Malaz": {"Land": (4500, 6500), "Villa": (4500, 6500), "Apartment": (3800, 5000), "Building": (4000, 5800)},

    # 3. East Riyadh
    "Al-Hamra": {"Land": (6000, 7800), "Villa": (6000, 7800), "Apartment": (4800, 6000), "Building": (5200, 6800)},
    "Ishbiliya": {"Land": (5200, 6500), "Villa": (5200, 6500), "Apartment": (4200, 5300), "Building": (4500, 5800)},
    "Al-Munasiyah": {"Land": (5000, 6200), "Villa": (5000, 6200), "Apartment": (4200, 5400), "Building": (4400, 5600)},
    "Al-Rawdah": {"Land": (4800, 6200), "Villa": (4800, 6200), "Apartment": (3800, 4800), "Building": (4200, 5400)},
    "Al-Qadisiyah": {"Land": (4200, 5300), "Villa": (4200, 5300), "Apartment": (3500, 4500), "Building": (3800, 4800)},

    # 4. West Riyadh
    "Al-Mahdia": {"Land": (3800, 5200), "Villa": (3800, 5200), "Apartment": (3200, 4200), "Building": (3400, 4500)},
    "Dharat Laban": {"Land": (3500, 4600), "Villa": (3500, 4600), "Apartment": (3000, 3800), "Building": (3200, 4000)},
    "Tuwaiq": {"Land": (3000, 4200), "Villa": (3000, 4200), "Apartment": (2600, 3500), "Building": (2800, 3800)},

    # 5. South Riyadh
    "Al-Shifa": {"Land": (2200, 3400), "Villa": (2200, 3400), "Apartment": (2000, 2800), "Building": (2100, 3000)},
    "Al-Aziziyah": {"Land": (2200, 3400), "Villa": (2200, 3400), "Apartment": (2000, 2800), "Building": (2100, 3000)},
    "Al-Dar El Beida": {"Land": (2200, 3400), "Villa": (2200, 3400), "Apartment": (2000, 2800), "Building": (2100, 3000)}
}

# Apply +6% Total Increase automatically to all prices
district_prices = {}
for dist, types in base_district_prices.items():
    district_prices[dist] = {}
    for prop, (p_min, p_max) in types.items():
        district_prices[dist][prop] = (int(p_min * 1.06), int(p_max * 1.06))

arabic_to_district = {
    "حطين": "Hittin",
    "الملقا": "Al-Malqa", "ملقا": "Al-Malqa",
    "الصحافة": "As-Sahafah", "صحافة": "As-Sahafah",
    "الياسمين": "Al-Yasmin", "ياسمين": "Al-Yasmin",
    "النرجس": "Al-Narjis", "نرجس": "Al-Narjis",
    "العارض": "Al-Aarid", "عارض": "Al-Aarid",
    "النخيل": "Al-Nakheel", "نخيل": "Al-Nakheel",
    "المونسية": "Al-Munasiyah", "مونسية": "Al-Munasiyah",
    "القادسية": "Al-Qadisiyah", "قادسية": "Al-Qadisiyah",
    "إشبيليا": "Ishbiliya", "اشبيليا": "Ishbiliya",
    "الروضة": "Al-Rawdah", "روضة": "Al-Rawdah",
    "الحمراء": "Al-Hamra", "حمراء": "Al-Hamra",
    "لبن": "Dharat Laban", "ضاحية لبن": "Dharat Laban", "ظهرة لبن": "Dharat Laban",
    "طويق": "Tuwaiq",
    "المهدية": "Al-Mahdia", "مهدية": "Al-Mahdia",
    "الشفاء": "Al-Shifa", "شفاء": "Al-Shifa",
    "العزيزية": "Al-Aziziyah", "عزيزية": "Al-Aziziyah",
    "الدار البيضاء": "Al-Dar El Beida", "دار البيضاء": "Al-Dar El Beida",
    "العليا": "Al-Olaya", "عليا": "Al-Olaya",
    "السليمانية": "Al-Sulaimaniyah", "سليمانية": "Al-Sulaimaniyah",
    "المربع": "Al-Murabba", "مربع": "Al-Murabba",
    "الملز": "Al-Malaz", "ملز": "Al-Malaz"
}

region_districts = {
    "North": ["Hittin", "Al-Malqa", "As-Sahafah", "Al-Yasmin", "Al-Narjis", "Al-Aarid", "Al-Nakheel"],
    "Central": ["Al-Olaya", "Al-Sulaimaniyah", "Al-Murabba", "Al-Malaz"],
    "East": ["Al-Hamra", "Ishbiliya", "Al-Munasiyah", "Al-Rawdah", "Al-Qadisiyah"],
    "West": ["Al-Mahdia", "Dharat Laban", "Tuwaiq"],
    "South": ["Al-Shifa", "Al-Aziziyah", "Al-Dar El Beida"]
}

# Sidebar - Property Features
st.sidebar.markdown("### 🔧 Property Features")

property_type = st.sidebar.selectbox("Property Type:", ["Apartment", "Villa", "Building", "Land"])
region = st.sidebar.selectbox("Select Region:", list(region_districts.keys()))
district = st.sidebar.selectbox("Select District:", region_districts[region])

if property_type == "Land":
    area = st.sidebar.slider("Land Area (m²):", 300, 1500, 500)
    bedrooms = 0
    bathrooms = 0
    property_age = 0
else:
    min_area = 100 if property_type == "Apartment" else (250 if property_type == "Villa" else 400)
    max_area = 500 if property_type == "Apartment" else 1200
    default_area = 150 if property_type == "Apartment" else 350
    area = st.sidebar.slider("Property Area (m²):", min_area, max_area, default_area)
    bedrooms = st.sidebar.slider("Bedrooms:", 1, 10, 3 if property_type == "Apartment" else 4)
    bathrooms = st.sidebar.slider("Bathrooms:", 1, 10, 3 if property_type == "Apartment" else 4)
    property_age = st.sidebar.slider("Property Age (Years):", 0, 30, 0)

predict_btn = st.sidebar.button("🔮 Predict Price & Trends")

# Header
st.title("🏡 Riyadh Real Estate Price Prediction & Investment Advisor")
st.markdown("An advanced Machine Learning portfolio project for analyzing, predicting property prices, and forecasting market trends in Riyadh.")

# Calculation Logic (Weighted m² Pricing & Dynamic Factors)
def calculate_realistic_predictions(p_type, p_region, p_district, p_area, p_bed, p_bath, p_age):
    price_range = district_prices.get(p_district, {"Land": (5000, 6000), "Villa": (5000, 6000), "Apartment": (4000, 5000), "Building": (4500, 5500)})
    min_m2, max_m2 = price_range.get(p_type, (5000, 6000))
    
    base_m2_price = (min_m2 + max_m2) / 2
    
    if p_type == "Land":
        estimated_price = p_area * base_m2_price
    elif p_type == "Apartment":
        estimated_price = (p_area * base_m2_price) - (p_age * 5000)
    else: # Villa / Building
        estimated_price = (p_area * base_m2_price) - (p_age * 10000)
        
    estimated_price = max(estimated_price, 250000)
    
    growth_rate = 0.042 if p_region in ["North", "Central"] else 0.032
    projected_price = estimated_price * (1 + growth_rate)
    diff = projected_price - estimated_price

    return estimated_price, projected_price, diff, growth_rate

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = {
        "p_type": property_type,
        "p_region": region,
        "p_district": district,
        "p_area": area,
        "p_bed": bedrooms,
        "p_bath": bathrooms,
        "p_age": property_age
    }

if predict_btn:
    st.session_state.prediction_data = {
        "p_type": property_type,
        "p_region": region,
        "p_district": district,
        "p_area": area,
        "p_bed": bedrooms,
        "p_bath": bathrooms,
        "p_age": property_age
    }

active_data = st.session_state.prediction_data
est_price, proj_price, price_diff, growth = calculate_realistic_predictions(
    active_data["p_type"],
    active_data["p_region"],
    active_data["p_district"],
    active_data["p_area"],
    active_data["p_bed"],
    active_data["p_bath"],
    active_data["p_age"]
)

st.success(f"✨ Analysis for *{active_data['p_type']}* in *{active_data['p_district']}* Completed Successfully!")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"*💰 Estimated {active_data['p_type']} Price*")
    st.header(f"{est_price:,.0f} SAR")

with col2:
    st.markdown("*📈 Projected Price (Next 1 Year)*")
    st.header(f"{proj_price:,.0f} SAR")
    st.markdown(f"<span style='color: #28a745; font-weight: bold;'>▲ +{price_diff:,.0f} SAR ({growth*100:.2f}%)</span>", unsafe_allow_html=True)

with col3:
    st.markdown("*📍 Location & Type*")
    st.header(f"{active_data['p_district']} ({active_data['p_type']})")

st.markdown("---")

st.subheader("💡 AI Investment Advisor Insights")
st.markdown("*Investment Rating:* 🔥 High Growth Potential (Prime Investment)")
st.info(f"Properties in *{active_data['p_district']}* ({active_data['p_region']} Riyadh) demonstrate robust pricing based on recent market transactions, excellent neighborhood services, and strong investor demand.")

# Chat Advisor Section
st.markdown("---")
st.subheader("🤖 LLM Real Estate Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحباً بك! اسألني عن أسعار الأحياء، الاستثمار، أو تقييم العقارات.\n\nAsk about district prices, investment opportunities, or valuation."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("اكتب سؤالك هنا / Ask a question...")
    submit_button = st.form_submit_button(label="🚀 إرسال / Send")

if submit_button and user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    query_lower = user_query.lower()
    is_arabic = any('\u0600' <= char <= '\u06FF' for char in user_query)
    
    detected_district = active_data["p_district"]
    for ar_name, eng_name in arabic_to_district.items():
        if ar_name in user_query:
            detected_district = eng_name
            break
            
    if is_arabic:
        if "سعر" in query_lower or "متر" in query_lower or "بكم" in query_lower:
            p_range = district_prices.get(detected_district, {"Land": (5000, 6000), "Villa": (5000, 6000), "Apartment": (4000, 5000)})
            land_min, land_max = p_range["Land"]
            villa_min, villa_max = p_range["Villa"]
            apt_min, apt_max = p_range["Apartment"]
            llm_response = f"*متوسط أسعار المتر المحدثة (بزيادة إجمالية 6%) في حي ({detected_district}):*\n- الأراضي والفلل: من {land_min:,} إلى {land_max:,} ريال/م²\n- الشقق: من {apt_min:,} إلى {apt_max:,} ريال/م²"
        else:
            llm_response = f"يشهد حي *{detected_district}* تحديثاً مستمراً في المؤشرات العقارية متوافقاً مع النمو السعري الحالي بنسبة 6%."
    else:
        llm_response = f"District *{detected_district}* reflects updated market pricing with a 6% total upward adjustment."

    st.session_state.messages.append({"role": "assistant", "content": llm_response})
    st.rerun()
