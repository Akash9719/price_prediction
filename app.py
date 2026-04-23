import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ================= LOAD =================
model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))
brand_avg_price = pickle.load(open("brand_avg_price.pkl","rb"))
model_freq = pickle.load(open("model_freq.pkl","rb"))

# ================= FEATURES =================
def create_features(df):
    df = df.copy()
    current_year = datetime.now().year

    df["car_age"] = np.maximum(0, current_year - df["year"])

    df["age"] = df["car_age"]
    df["age_squared"] = df["car_age"] ** 2
    df["age_log"] = np.log1p(df["car_age"])

    df["kms_log"] = np.log1p(df["kms"])
    df["kms_per_year"] = df["kms"] / (df["car_age"] + 1)
    df["km_per_year"] = df["kms"] / (df["car_age"] + 1)

    df["age_kms"] = df["car_age"] * df["kms"]
    df["age_kms_ratio"] = df["car_age"] / (df["kms"] + 1)

    df["depreciation_curve"] = np.exp(-df["car_age"] / 6)

    df["kms_per_owner"] = df["kms"] / (df["owners"] + 1)
    df["age_per_owner"] = df["car_age"] / (df["owners"] + 1)

    return df

# ================= DEPRECIATION =================
def depreciation_factor(age, brand):
    brand = str(brand).lower()

    if "maruti" in brand or "hyundai" in brand:
        curve = "economy"
    elif "bmw" in brand or "mercedes" in brand or "audi" in brand:
        curve = "luxury"
    else:
        curve = "standard"

    if curve == "economy":
        if age <= 1:
            return 0.92
        elif age <= 3:
            return 0.85 - 0.04 * (age - 1)
        elif age <= 6:
            return 0.73 - 0.05 * (age - 3)
        elif age <= 10:
            return 0.58 - 0.05 * (age - 6)
        elif age <= 15:
            return 0.38 - 0.04 * (age - 10)
        else:
            return 0.15

    elif curve == "luxury":
        if age <= 1:
            return 0.85
        elif age <= 3:
            return 0.70 - 0.07 * (age - 1)
        elif age <= 6:
            return 0.50 - 0.06 * (age - 3)
        elif age <= 10:
            return 0.32 - 0.05 * (age - 6)
        elif age <= 15:
            return 0.15 - 0.03 * (age - 10)
        else:
            return 0.08

    else:
        if age <= 1:
            return 0.90
        elif age <= 3:
            return 0.80 - 0.05 * (age - 1)
        elif age <= 6:
            return 0.65 - 0.05 * (age - 3)
        elif age <= 10:
            return 0.50 - 0.04 * (age - 6)
        elif age <= 15:
            return 0.30 - 0.03 * (age - 10)
        else:
            return 0.10


def final_price(pred_price, age, brand):
    base = pred_price * depreciation_factor(age, brand)

    if age > 5:
        base *= 0.95
    if age > 10:
        base *= 0.90

    return base


def apply_scrap_floor(price, age):
    scrap_value = 50000
    if age > 15:
        return max(price, scrap_value)
    return price

# ================= UI =================
st.title("🚗 Car Price Predictor")

year = st.number_input("Year", 2000, 2025, 2020)
kms = st.number_input("Kms", 0, 200000, 30000)
owners = st.selectbox("Owners", [1,2,3,4])
city = st.text_input("City", "Delhi")
fuel = st.selectbox("Fuel", ["Petrol","Diesel","CNG"])
brand = st.text_input("Brand", "Maruti Suzuki")
model_clean = st.text_input("Model", "Swift")

# ================= PREDICT =================
if st.button("Predict Price"):

    input_df = pd.DataFrame([{
        "year": year,
        "kms": kms,
        "owners": owners,
        "city": city,
        "fuel": fuel,
        "brand": brand,
        "model_clean": model_clean
    }])

    # Feature Engineering
    input_df = create_features(input_df)

    # Capture before reindex
    age = input_df["age"].iloc[0]
    brand_val = input_df["brand"].iloc[0]

    # Brand encoding (compressed)
    bv = input_df["brand"].map(brand_avg_price).fillna(0.5)
    input_df["brand_value"] = np.clip(np.log1p(bv) / 10, 0, 2)

    # Model frequency
    input_df["model_freq"] = input_df["model_clean"].map(model_freq).fillna(0)

    # Align columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction (log scale)
    pred_ratio = model.predict(input_df)[0]
    approx_new_price = input_df["approx_new_price"].iloc[0]
    pred_price = pred_ratio * approx_new_price

    # Apply corrections
    pred_price = final_price(pred_price, age, brand_val)
    pred_price = apply_scrap_floor(pred_price, age)

    # Output
    st.success(f"💰 Estimated Price: ₹ {int(pred_price):,}")
