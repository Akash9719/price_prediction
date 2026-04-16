import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# -------------------------------
# Load model and columns (cached)
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    cols = pickle.load(open("columns.pkl", "rb"))
    return model, cols

model, cols = load_artifacts()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to estimate price")

# -------------------------------
# Extract dropdown values dynamically
# -------------------------------
cities = sorted([col.replace("city_", "") for col in cols if col.startswith("city_")])
fuels = sorted([col.replace("fuel_", "") for col in cols if col.startswith("fuel_")])
brands = sorted([col.replace("brand_", "") for col in cols if col.startswith("brand_")])
models = sorted([col.replace("model_clean_", "") for col in cols if col.startswith("model_clean_")])

# -------------------------------
# User Inputs
# -------------------------------
kms = st.number_input("Kilometers Driven", min_value=0, value=50000)
owners = st.selectbox("Number of Owners", [0, 1, 2, 3])
year = st.number_input("Manufacturing Year", 1990, datetime.now().year, value=2018)

city = st.selectbox("City", cities)
fuel = st.selectbox("Fuel Type", fuels)
brand = st.selectbox("Brand", brands)
model_name = st.selectbox("Model", models)

# -------------------------------
# Derived features
# -------------------------------
car_age = datetime.now().year - year
kms_per_year = kms / car_age if car_age > 0 else kms

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):

    # 🔥 CORRECT way to create input dataframe
    input_df = pd.DataFrame(0, index=[0], columns=cols)

    # numeric features
    input_df["kms"] = kms
    input_df["owners"] = owners
    input_df["year"] = year
    input_df["car_age"] = car_age
    input_df["kms_per_year"] = kms_per_year

    # categorical (one-hot)
    input_df[f"city_{city}"] = 1
    input_df[f"fuel_{fuel}"] = 1
    input_df[f"brand_{brand}"] = 1
    input_df[f"model_clean_{model_name}"] = 1

    # ensure correct dtype
    input_df = input_df.astype(float)

    # -------------------------------
    # Prediction
    # -------------------------------
    try:
        prediction = model.predict(input_df)[0]

        # nice UI output
        st.metric("💰 Estimated Price", f"₹ {round(prediction, 2):,}")

        # optional debug (remove later)
        with st.expander("See input features"):
            st.write(input_df)

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")

# -------------------------------
# Footer
# -------------------------------
st.caption("⚠️ Price is an estimate based on historical data and may vary.")