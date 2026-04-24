import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# -------------------------------
# Load artifacts
# -------------------------------
pipeline = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
brand_avg_price = pickle.load(open("brand_avg_price.pkl", "rb"))
model_freq_dict = pickle.load(open("model_freq.pkl", "rb"))

# Normalize keys
brand_avg_price = {k.lower(): v for k, v in brand_avg_price.items()}
model_freq_dict = {k.lower(): v for k, v in model_freq_dict.items()}

# -------------------------------
# Approx price function (SMART)
# -------------------------------
def get_approx_new_price(brand, model):
    brand_key = brand.lower()
    base_price = brand_avg_price.get(
        brand_key,
        np.mean(list(brand_avg_price.values()))
    )

    # Stable variation (important)
    seed = abs(hash(model)) % 100
    variation = 0.9 + (seed / 1000)  # ~0.9–1.0

    return base_price * variation

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.markdown("""
# 🚗 Car Price Prediction
### AI-powered resale valuation
""")

# -------------------------------
# Sidebar Inputs (PRO UI)
# -------------------------------
st.sidebar.header("🔧 Enter Car Details")

current_year = datetime.now().year

year = st.sidebar.number_input("Year", 2000, current_year, 2018)
kms = st.sidebar.number_input("Kilometers Driven", 0, 300000, 50000)
owner = st.sidebar.selectbox("Owner", [1,2,3,4])

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol","Diesel","CNG","Electric"])

brand = st.sidebar.selectbox("Brand", [
    "Maruti Suzuki","Hyundai","Tata","Mahindra","Kia",
    "Honda","Toyota","Renault","Skoda","Volkswagen","MG",
    "BMW","Mercedes-Benz","Audi","Jaguar","Land Rover","Volvo"
])

model = st.sidebar.text_input("Model (type manually)", "Swift")

state = st.sidebar.selectbox("State", [
    "Delhi","Uttar Pradesh","Maharashtra","Karnataka","Tamil Nadu",
    "Gujarat","Rajasthan","Punjab","Haryana","West Bengal"
])

# -------------------------------
# Predict Button
# -------------------------------
st.markdown("---")

if st.sidebar.button("💰 Predict Price"):

    try:
        # -------------------------------
        # Feature Engineering
        # -------------------------------
        age = current_year - year

        km_per_year = kms / age if age > 0 else kms
        age_kms = age * kms
        age_kms_ratio = kms / age if age > 0 else kms

        kms_log = np.log1p(kms)
        age_log = np.log1p(age)
        age_squared = age ** 2
        depreciation_curve = age * 0.1

        model_key = model.lower().strip()
        brand_key = brand.lower().strip()

        brand_value = brand_avg_price.get(
            brand_key,
            np.mean(list(brand_avg_price.values()))
        )

        model_freq = model_freq_dict.get(
            model_key,
            np.mean(list(model_freq_dict.values()))
        )

        # -------------------------------
        # Input Data
        # -------------------------------
        input_data = pd.DataFrame([{
            "year": year,
            "kms": kms,
            "owners": owner,
            "city": state,
            "brand": brand,
            "model_clean": model_key,
            "fuel": fuel,

            "age": age,
            "age_squared": age_squared,
            "age_log": age_log,
            "kms_log": kms_log,
            "km_per_year": km_per_year,
            "age_kms": age_kms,
            "age_kms_ratio": age_kms_ratio,
            "depreciation_curve": depreciation_curve,
            "brand_value": brand_value,
            "model_freq": model_freq
        }])

        input_data = input_data[columns]

        # -------------------------------
        # Predict Ratio
        # -------------------------------
        pred_ratio = pipeline.predict(input_data)[0]

        # -------------------------------
        # Convert to Price
        # -------------------------------
        approx_price = get_approx_new_price(brand, model)
        predicted_price = pred_ratio * approx_price

        # -------------------------------
        # Confidence Range
        # -------------------------------
        lower = predicted_price * 0.9
        upper = predicted_price * 1.1

        # -------------------------------
        # Display
        # -------------------------------
        st.success("✅ Prediction Generated")

        st.markdown(f"""
        ## 💰 Estimated Price
        # ₹ {int(predicted_price):,}
        """)

        st.info(f"📊 Expected Range: ₹ {int(lower):,} – ₹ {int(upper):,}")

        # Confidence indicator
        confidence = min(max(pred_ratio, 0), 1)

        st.progress(float(confidence))
        st.caption(f"Model Confidence Score: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
