import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ================= LOAD =================
model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))
brand_avg_price = pickle.load(open("brand_avg_price.pkl","rb"))
global_mean_price = pickle.load(open("global_mean_price.pkl","rb"))
model_freq = pickle.load(open("model_freq.pkl","rb"))

# ================= FEATURES =================
def create_features(df):
    df = df.copy()
    current_year = datetime.now().year

    df["car_age"] = np.maximum(0, current_year - df["year"])
    df["car_age_squared"] = df["car_age"] ** 2
    df["car_age_cube"] = df["car_age"] ** 3

    df["kms_log"] = np.log1p(df["kms"])
    df["kms_per_year"] = df["kms"] / (df["car_age"] + 1)

    df["age_kms"] = df["car_age"] * df["kms"]
    df["age_kms_ratio"] = df["car_age"] / (df["kms"] + 1)

    df["depreciation_curve"] = np.exp(-df["car_age"] / 6)

    df["kms_per_owner"] = df["kms"] / (df["owners"] + 1)
    df["age_per_owner"] = df["car_age"] / (df["owners"] + 1)

    return df

# ================= UI =================
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

    # Features
    input_df = create_features(input_df)

    # Brand value
    input_df["brand_value"] = input_df["brand"].map(brand_avg_price).fillna(global_mean_price)

    # Model freq
    input_df["model_freq"] = input_df["model_clean"].map(model_freq).fillna(0)

    # DEBUG CHECK
    missing_cols = set(columns) - set(input_df.columns)
    st.write("Missing columns:", missing_cols)

    # Align columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Predict
    pred_log = model.predict(input_df)[0]
    prediction = int(np.expm1(pred_log))

    st.success(f"Estimated Price: ₹ {prediction:,}")
