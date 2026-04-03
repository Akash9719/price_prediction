import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
cols = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to estimate price")

# -------------------------
# Extract dropdown values dynamically
# -------------------------

cities = sorted([col.replace("city_", "") for col in cols if col.startswith("city_")])
fuels = sorted([col.replace("fuel_", "") for col in cols if col.startswith("fuel_")])
brands = sorted([col.replace("brand_", "") for col in cols if col.startswith("brand_")])
models = sorted([col.replace("model_clean_", "") for col in cols if col.startswith("model_clean_")])

# -------------------------
# User Inputs
# -------------------------

kms = st.number_input("Kilometers Driven", min_value=0)
owners = st.selectbox("Number of Owners", [0, 1, 2, 3])
year = st.number_input("Manufacturing Year", 1990, datetime.now().year)

city = st.selectbox("City", cities)
fuel = st.selectbox("Fuel Type", fuels)
brand = st.selectbox("Brand", brands)
model_name = st.selectbox("Model", models)

# Derived features
car_age = datetime.now().year - year
kms_per_year = kms / car_age if car_age > 0 else kms

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Price"):

    input_dict = {
        "kms": kms,
        "owners": owners,
        "year": year,
        "car_age": car_age,
        "kms_per_year": kms_per_year,
        
        f"city_{city}": 1,
        f"fuel_{fuel}": 1,
        f"brand_{brand}": 1,
        f"model_clean_{model_name}": 1
    }

    # Create dataframe with all columns
    input_df = pd.DataFrame(columns=cols)

    for col in input_df.columns:
        input_df.loc[0, col] = input_dict.get(col, 0)

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)}")