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

# Normalize dict keys
brand_avg_price = {k.lower(): v for k, v in brand_avg_price.items()}
model_freq_dict = {k.lower(): v for k, v in model_freq_dict.items()}

# -------------------------------
# Brand mapping (CRITICAL)
# -------------------------------
brand_mapping = {
    "Maruti Suzuki": "maruti",
    "Hyundai": "hyundai",
    "Tata": "tata",
    "Mahindra": "mahindra",
    "Kia": "kia",
    "Honda": "honda",
    "Toyota": "toyota",
    "Renault": "renault",
    "Skoda": "skoda",
    "Volkswagen": "volkswagen",
    "MG": "mg",
    "BMW": "bmw",
    "Mercedes-Benz": "mercedes",
    "Audi": "audi",
    "Jaguar": "jaguar",
    "Land Rover": "land rover",
    "Volvo": "volvo"
}

# -------------------------------
# UI Data
# -------------------------------
brand_model_map = {
    "Maruti Suzuki": ["Swift", "Baleno", "WagonR", "Alto", "Dzire", "Brezza"],
    "Hyundai": ["i10", "i20", "Creta", "Verna", "Venue"],
    "Tata": ["Nexon", "Harrier", "Safari", "Tiago", "Altroz"],
    "Mahindra": ["Scorpio", "XUV700", "Thar"],
    "Kia": ["Seltos", "Sonet"],
    "Honda": ["City", "Amaze"],
    "Toyota": ["Innova", "Fortuner"],
    "Renault": ["Kwid", "Kiger"],
    "Skoda": ["Rapid", "Kushaq"],
    "Volkswagen": ["Polo", "Taigun"],
    "MG": ["Hector", "Astor"],
    "BMW": ["X1", "X3"],
    "Mercedes-Benz": ["C-Class", "E-Class"],
    "Audi": ["A4", "Q5"],
    "Jaguar": ["XF"],
    "Land Rover": ["Defender"],
    "Volvo": ["XC60"]
}

states = [
    "Delhi","Uttar Pradesh","Maharashtra","Karnataka","Tamil Nadu",
    "Gujarat","Rajasthan","Punjab","Haryana","West Bengal"
]

fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]
owners = [1, 2, 3, 4]
brands = list(brand_model_map.keys())

# -------------------------------
# UI Layout
# -------------------------------
st.title("🚗 Car Price Prediction")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", 2000, datetime.now().year, 2018)
    kms = st.number_input("KMs Driven", 0, 300000, 50000)
    owner = st.selectbox("Owner", owners)
    state = st.selectbox("State", states)

with col2:
    brand = st.selectbox("Brand", brands)
    model = st.selectbox("Model", brand_model_map[brand])
    fuel = st.selectbox("Fuel", fuel_types)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):

    try:
        current_year = datetime.now().year
        age = current_year - year

        # Feature Engineering (MATCH TRAINING)
        km_per_year = kms / age if age > 0 else kms
        age_kms = age * kms
        age_kms_ratio = kms / age if age > 0 else kms

        kms_log = np.log1p(kms)
        age_log = np.log1p(age)
        age_squared = age ** 2
        depreciation_curve = age * 0.1

        # FIXED MAPPING
        brand_key = brand_mapping.get(brand, brand.lower())
        model_key = model.lower().strip()

        brand_value = brand_avg_price.get(
            brand_key,
            np.mean(list(brand_avg_price.values()))
        )

        model_freq = model_freq_dict.get(
            model_key,
            np.mean(list(model_freq_dict.values()))
        )

        # Final input
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

        # Align columns
        input_data = input_data[columns]

        pred_ratio = pipeline.predict(input_data)[0]
        brand_key = brand.lower()
        approx_price = brand_avg_price.get(
            brand_key,
            np.mean(list(brand_avg_price.values()))
        )

        prediction = pred_ratio * approx_price

        st.success(f"Estimated Price: ₹ {int(prediction):,}")

    except Exception as e:
        st.error(f"Error: {e}")
