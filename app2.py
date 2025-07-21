import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils import preprocess_input

# Load CatBoost model
try:
    model = pickle.load(open('model/CAT_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Make sure it’s in the `model/` folder.")
    st.stop()

# Page config
st.set_page_config(page_title="📊 Manual Forecasting", layout="centered")
st.title("🧮 Forecast Product Demand (Manual Input)")

# --- INPUT SECTION ---
st.markdown("### 📥 Enter product & store details:")

date = st.date_input("Date")
family = st.selectbox("Product Family", ['BEVERAGES', 'GROCERY I', 'DAIRY', 'SEAFOOD', 'HOME AND KITCHEN I'])
city = st.selectbox("City", ['QUITO', 'CUENCA', 'GUAYAQUIL', 'AMBATO'])
holiday_type = st.selectbox("Holiday Type", ['Holiday', 'Additional', 'Event', 'Work Day', 'Transfer'])
transactions = st.number_input("Number of Transactions", min_value=0, step=1)
oil_price = st.number_input("Oil Price (dcoilwtico)", min_value=0.0, step=0.1)

# --- PREPROCESS AND PREDICT ---
if st.button("Predict Sales"):
    try:
        # Construct DataFrame
        input_dict = {
            'id': [1],
            'date': [pd.to_datetime(date)],
            'family': [family],
            'city': [city],
            'holiday_type': [holiday_type],
            'transactions': [transactions],
            'dcoilwtico': [oil_price]
        }
        input_df = pd.DataFrame(input_dict)

        st.subheader("🔎 Input Summary")
        st.dataframe(input_df)

        # Preprocess and predict
        processed_df = preprocess_input(input_df)
        prediction = model.predict(processed_df)[0]

        st.success(f"📈 Predicted Sales: **{prediction:.2f}** units")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
