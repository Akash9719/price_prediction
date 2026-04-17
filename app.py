import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load model and columns
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    return model

model = load_artifacts()

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to estimate price")

# -------------------------
# Create mapping (CRITICAL FIX)
# -------------------------
city_map = {col.replace("city_", ""): col for col in cols if col.startswith("city_")}
fuel_map = {col.replace("fuel_", ""): col for col in cols if col.startswith("fuel_")}
brand_map = {col.replace("brand_", ""): col for col in cols if col.startswith("brand_")}
model_map = {col.replace("model_clean_", ""): col for col in cols if col.startswith("model_clean_")}

# -------------------------
# Dropdowns
# -------------------------
kms = st.number_input("Kilometers Driven", min_value=0, value=50000)
owners = st.selectbox("Number of Owners", [0, 1, 2, 3])
year = st.number_input("Manufacturing Year", 1990, datetime.now().year, value=2018)

city = st.selectbox("City", sorted(city_map.keys()))
fuel = st.selectbox("Fuel Type", sorted(fuel_map.keys()))
brand = st.selectbox("Brand", sorted(brand_map.keys()))
model_name = st.selectbox("Model", sorted(model_map.keys()))

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price"):

    # Create input dataframe correctly
    input_df = pd.DataFrame(0, index=[0], columns=cols)

    # numeric
    input_df["kms"] = kms
    input_df["owners"] = owners
    input_df["year"] = year
   
    # categorical (SAFE mapping)
    input_df[city_map[city]] = 1
    input_df[fuel_map[fuel]] = 1
    input_df[brand_map[brand]] = 1
    input_df[model_map[model_name]] = 1

    # ensure float
    input_df = input_df.astype(float)

    # Prediction
    try:
        prediction = model.predict(input_df)[0]

        st.metric("💰 Estimated Price", f"₹ {round(prediction, 2):,}")

        # Debug (optional)
        with st.expander("See input features"):
            st.write(input_df)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.caption("⚠️ Price is an estimate based on historical data.")
