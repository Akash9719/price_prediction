import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------------
# Load model and columns
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
cols = pickle.load(open("columns.pkl", "rb"))

# -------------------------------
# UI Title
# -------------------------------
st.title("🚗 Car Price Prediction")

# -------------------------------
# User Inputs
# -------------------------------
kms = st.number_input("Kilometers Driven", min_value=0, value=50000)
owners = st.selectbox("Number of Owners", [1, 2, 3, 4])
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2018)

city = st.selectbox("City", ["Ahmedabad", "Bangalore", "Chennai", "Delhi"])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
brand = st.selectbox("Brand", ["Hyundai", "Maruti", "Honda", "Toyota"])
model_name = st.selectbox("Model", ["i20", "Swift", "City", "Innova"])

# -------------------------------
# Feature Engineering
# -------------------------------
car_age = 2024 - year
kms_per_year = kms / car_age if car_age > 0 else kms

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Price"):

    # Create input dictionary
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

    # Create dataframe with same columns as training
    input_df = pd.DataFrame(columns=cols)

    for col in input_df.columns:
        input_df.loc[0, col] = input_dict.get(col, 0)

    # -------------------------------
    # VERY IMPORTANT FIXES
    # -------------------------------
    input_df = input_df.astype(float)
    input_df = input_df.fillna(0)
    input_df = input_df.reindex(columns=cols, fill_value=0)

    # -------------------------------
    # Prediction
    # -------------------------------
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)}")

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
