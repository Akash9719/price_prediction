import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# -------------------------------
# Load model (pipeline)
# -------------------------------
pipeline = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction App")
st.markdown("### Smart resale price prediction using ML")

# -------------------------------
# Brand → Model Mapping
# -------------------------------
brand_model_map = {

    # Mass
    "Maruti Suzuki": ["Swift", "Baleno", "WagonR", "Alto", "Dzire", "Brezza", "Ertiga", "Ciaz"],
    "Hyundai": ["i10", "i20", "Creta", "Verna", "Venue", "Aura", "Alcazar"],
    "Tata": ["Nexon", "Harrier", "Safari", "Tiago", "Tigor", "Altroz", "Punch"],
    "Mahindra": ["Scorpio", "XUV300", "XUV700", "Bolero", "Thar"],
    "Kia": ["Seltos", "Sonet", "Carens"],
    "Honda": ["City", "Amaze", "WR-V", "Jazz"],
    "Toyota": ["Innova", "Fortuner", "Glanza", "Urban Cruiser", "Camry"],
    "Renault": ["Kwid", "Triber", "Kiger", "Duster"],
    "Skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia"],
    "Volkswagen": ["Polo", "Vento", "Virtus", "Taigun", "Tiguan"],
    "MG": ["Hector", "Astor", "ZS EV", "Gloster"],

    # Luxury
    "BMW": ["X1", "X3", "X5", "3 Series", "5 Series"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLA", "GLC"],
    "Audi": ["A4", "A6", "Q3", "Q5"],
    "Jaguar": ["XE", "XF", "F-Pace"],
    "Land Rover": ["Defender", "Discovery", "Range Rover Evoque"],
    "Volvo": ["XC40", "XC60", "XC90"]
}

# -------------------------------
# States & UTs
# -------------------------------
states = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand",
    "Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Punjab",
    "Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal",
    "Delhi","Jammu & Kashmir","Ladakh","Chandigarh",
    "Andaman & Nicobar Islands","Dadra & Nagar Haveli and Daman & Diu",
    "Lakshadweep","Puducherry"
]

fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]
owners = [1, 2, 3, 4]
brands = list(brand_model_map.keys())

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    current_year = datetime.now().year

    year = st.number_input("📅 Year", 2000, current_year, 2018)
    kms = st.number_input("🚘 Kilometers Driven", 0, 300000, 50000)
    owner = st.selectbox("👤 Owner", owners)
    state = st.selectbox("🌍 State / UT", states)

with col2:
    brand = st.selectbox("🏷️ Brand", brands)
    model = st.selectbox("🚗 Model", brand_model_map[brand])
    fuel = st.selectbox("⛽ Fuel Type", fuel_types)

# -------------------------------
# Prediction
# -------------------------------
st.markdown("---")

if st.button("💰 Predict Price"):

    try:
        # ONLY RAW INPUTS (IMPORTANT FIX)
        input_data = pd.DataFrame([{
            "year": year,
            "kms": kms,
            "owners": owner,
            "city": state,
            "brand": brand,
            "model": model,
            "fuel": fuel
        }])

        # Debug (optional)
        # st.write(input_data)

        prediction = pipeline.predict(input_data)[0]

        st.success("✅ Prediction Successful!")

        st.markdown(
            f"""
            <div style="padding:20px;border-radius:10px;background:#f0f2f6">
                <h2 style="color:green;">💸 Estimated Price</h2>
                <h1>₹ {int(prediction):,}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
