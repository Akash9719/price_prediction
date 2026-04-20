import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to estimate price")

# -------------------------
# Inputs
# -------------------------
kms = st.number_input("Kilometers Driven", min_value=0, value=50000)
owners = st.selectbox("Number of Owners", [0, 1, 2, 3])
year = st.number_input("Manufacturing Year", 1990, datetime.now().year, value=2018)

city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Ahmedabad"])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
brand = st.selectbox("Brand", ["Maruti Suzuki", "Hyundai", "Honda", "Toyota"])
model_name = st.text_input("Model (e.g., Swift, i20, City)")

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price"):

    current_year = datetime.now().year
    car_age = current_year - year
    kms_per_year = kms / (car_age + 1)
    age_weight = np.exp(-car_age / 5)

    # Feature Engineering (same as training)
    age_log = np.log1p(car_age)
    age_squared = car_age ** 2
    age_kms = car_age * kms

    input_df = pd.DataFrame([{
        "kms": kms,
        "owners": owners,
        "year": year,
        "car_age": car_age,
        "kms_per_year": kms_per_year,

        "age_log": age_log,
        "age_squared": age_squared,
        "age_kms": age_kms,
        "age_weight": age_weight,

        "city": city,
        "fuel": fuel,
        "brand": brand,
        "model_clean": model_name
    }])

    try:
        # 🔥 Raw model prediction
        prediction = np.expm1(model.predict(input_df))[0]

        # 🔥 Practical correction (VERY IMPORTANT)
        prediction = prediction * 0.65

        st.metric("💰 Estimated Price", f"₹ {round(prediction):,}")

        with st.expander("See input features"):
            st.write(input_df)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.caption("⚠️ Price is an estimate based on historical data.")
