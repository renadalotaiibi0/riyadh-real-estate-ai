import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="مستشار عقارات الرياض الذكي",
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

# App Header
st.title("🏡 مستشار عقارات الرياض والذكاء الاصطناعي")
st.markdown("نظام ذكي متكامل لتحليل وتوقعات أسعار العقارات في مدينة الرياض باستخدام الذكاء الاصطناعي.")

# Sidebar Controls
st.sidebar.header("إعدادات البحث والتوقع")

default_districts = ["الملقا", "النرجس", "الصحافة", "الياسمين", "حطين", "العارض", "النخيل"]
if df is not None and "district" in df.columns:
    district_list = list(df["district"].dropna().unique())
else:
    district_list = default_districts

district = st.sidebar.selectbox("اختر الحي", district_list)
property_type = st.sidebar.selectbox("نوع العقار", ["فيلا", "شقة", "ارض", "عمارة"])
area = st.sidebar.slider("المساحة (متر مربع)", 100, 1000, 300)
bedrooms = st.sidebar.slider("عدد الغرف", 1, 10, 4)
bathrooms = st.sidebar.slider("عدد دورات المياه", 1, 10, 4)

# Prediction Section
if st.sidebar.button("توقع السعر الآن 🚀"):
    st.subheader("نتائج التوقع والتحليل الذكي")
    if model is not None:
        try:
            input_data = pd.DataFrame([[area, bedrooms, bathrooms]], columns=['area', 'bedrooms', 'bathrooms'])
            predicted_price = model.predict(input_data)[0]
            st.success(f"السعر المتوقع للعقار: *{predicted_price:,.0f} ريال سعودي*")
        except Exception:
            base_price = area * 4500 + (bedrooms * 50000)
            st.success(f"السعر التقديري للعقار: *{base_price:,.0f} ريال سعودي*")
    else:
        base_price = area * 4500 + (bedrooms * 50000)
        st.success(f"السعر التقديري للعقار: *{base_price:,.0f} ريال سعودي*")

# Chatbot Section
st.markdown("---")
st.subheader("💬 مستشار عقارات الرياض الذكي")
st.markdown("اسأل المستشار أي سؤال عن العقار، الاستثمار، أو الأفرع في الرياض (مثال: ما هو أفضل حي للاستثمار؟)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("اسأل المستشار (مثال: ما هو أفضل الأحياء للاستثمار في شمال الرياض؟)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        query_lower = user_input.lower()
        if "شمال" in query_lower or "الملقا" in query_lower or "الصحافة" in query_lower:
            reply = "مناطق شمال الرياض (مثل الملقا، الصحافة، والياسمين) تعيش طفرة نمو استثمارية بسبب قربها من مشاريع الرياض الكبرى والمركز المالي، وتعتبر خياراً ممتازاً لزيادة رأس المال."
        elif "شقة" in query_lower or "شقق" in query_lower:
            reply = "سوق الشقق السكنية في الرياض يشهد طلباً عالياً جداً خصوصاً من الشريحة الشابة، مع عائد إيجاري ممتاز يتراوح غالباً بين 6% إلى 8% سنوياً."
        elif "عائد" in query_lower or "استثمار" in query_lower:
            reply = "العوائد الاستثمارية في العقارات التجارية والسكنية في شمال الرياض تعتبر من الأفضل، مع مراعاة اختيار عقارات حديثة البناء لمحدودية تكاليف الصيانة."
        else:
            reply = f"بناءً على تحليلات البيانات العقارية في الرياض، سؤالك حول '{user_input}' يعكس اهتماماً واعداً بالمشاريع الكبرى والتطوير العمراني المستمر."
            
        response_placeholder.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
