import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# -------------------------------
# Load model & artifacts (UNCHANGED)
# -------------------------------
pipeline = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction App")
st.markdown("### Predict resale value with smart ML")

# -------------------------------
# Brand → Model Mapping (COMPLETE)
# -------------------------------
brand_model_map = {

    # Mass Brands
    "Maruti Suzuki": ["Swift", "Baleno", "WagonR", "Alto", "Dzire", "Brezza", "Ertiga", "Ciaz"],
    "Hyundai": ["i10", "i20", "Creta", "Verna", "Venue", "Aura", "Alcazar"],
    "Tata": ["Nexon", "Harrier", "Safari", "Tiago", "Tigor", "Altroz", "Punch"],
    "Mahindra": ["Scorpio", "XUV300", "XUV700", "Bolero", "Thar", "Marazzo"],
    "Kia": ["Seltos", "Sonet", "Carens", "EV6"],
    "Honda": ["City", "Amaze", "WR-V", "Jazz", "Elevate"],
    "Toyota": ["Innova", "Fortuner", "Glanza", "Urban Cruiser", "Hyryder", "Camry"],
    "Renault": ["Kwid", "Triber", "Kiger", "Duster"],
    "Skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia"],
    "Volkswagen": ["Polo", "Vento", "Virtus", "Taigun", "Tiguan"],
    "MG": ["Hector", "Astor", "ZS EV", "Gloster", "Comet EV"],

    # Luxury Brands
    "BMW": ["X1", "X3", "X5", "3 Series", "5 Series", "7 Series"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLA", "GLC", "GLE"],
    "Audi": ["A3", "A4", "A6", "Q3", "Q5", "Q7"],
    "Jaguar": ["XE", "XF", "F-Pace", "I-Pace"],
    "Land Rover": ["Defender", "Discovery", "Range Rover Evoque", "Range Rover Sport"],
}

# -------------------------------
# Complete Indian States & UTs
# -------------------------------
states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan","Uttar Pradesh", "Uttarakhand", "West Bengal",

    # Union Territories
    "Delhi", "Jammu & Kashmir", "Ladakh", "Chandigarh",
    "Andaman & Nicobar Islands", "Dadra & Nagar Haveli and Daman & Diu",
    "Lakshadweep", "Puducherry"
]

fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]
owners = [1, 2, 3, 4]
brands = list(brand_model_map.keys())

# -------------------------------
# Layout (Clean UI)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    # Year Input
    current_year = datetime.now().year
    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=2000,
        max_value=current_year,
        value=2018,
        step=1
    )

    # KMs Driven
    kms = st.number_input(
        "🚘 Kilometers Driven",
        min_value=0,
        max_value=300000,
        value=50000,
        step=1000
    )

    # Owner
    owner = st.selectbox("👤 Owner Number", owners)

    # State
    state = st.selectbox("🌍 State / UT", states)

with col2:
    # Brand
    brand = st.selectbox("🏷️ Brand", brands)

    # Dynamic Model Dropdown
    model = st.selectbox(
        "🚗 Model",
        brand_model_map.get(brand, [])
    )

    # Fuel Type
    fuel = st.selectbox("⛽ Fuel Type", fuel_types)

# -------------------------------
# Predict Button
# -------------------------------
# -------------------------------
# Predict Button
# -------------------------------
st.markdown("---")

if st.button("💰 Predict Price"):

    if any(v is None for v in [year, kms, owner, state, brand, model, fuel]):
        st.warning("⚠️ Please fill all fields before predicting")
    else:
        try:
            # -------------------------------
            # Feature Engineering (CRITICAL FIX)
            # -------------------------------
            current_year = datetime.now().year
            age = current_year - year

            # Avoid division errors
            km_per_year = kms / age if age > 0 else kms
            age_kms = age * kms
            age_kms_ratio = kms / age if age > 0 else kms

            # Logs (safe)
            import numpy as np
            kms_log = np.log1p(kms)
            age_log = np.log1p(age)

            # Placeholder engineered features (safe defaults)
            age_squared = age ** 2
            depreciation_curve = age * 0.1  # simple proxy
            brand_value = 1  # placeholder (if you had mapping, load it)
            model_freq = 1   # placeholder
            model_clean = model.lower()

            # -------------------------------
            # Final Input DataFrame
            # -------------------------------
            input_data = pd.DataFrame([{
                "year": year,
                "kms": kms,
                "owners": owner,
                "city": state,
                "brand": brand,
                "model": model,
                "model_clean": model_clean,
                "fuel": fuel,

                # Engineered Features
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

            # -------------------------------
            # Align Columns (VERY IMPORTANT)
            # -------------------------------
            final_input = pd.DataFrame(columns=columns)
            for col in input_data.columns:
                if col in final_input.columns:
                    final_input[col] = input_data[col]
            final_input = final_input.fillna(0)

            # -------------------------------
            # Prediction
            # -------------------------------
            prediction = pipeline.predict(input_data)[0]

            # -------------------------------
            # Output
            # -------------------------------
            st.success("✅ Prediction Successful!")

            st.markdown(
                f"""
                <div style="padding:20px;border-radius:10px;background-color:#f0f2f6">
                    <h2 style="color:green;">💸 Estimated Price</h2>
                    <h1>₹ {int(prediction):,}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
