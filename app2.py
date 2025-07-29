import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained XGBoost model
with open("XGB_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved feature names used during training
with open("model_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Set Streamlit app title
st.title("📊 Product Demand Forecasting")
st.markdown("Enter the details below to predict future sales.")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        store_nbr = st.number_input("Store Number", value=1, min_value=1)
        onpromotion = st.number_input("Items on Promotion", value=0)
        cluster = st.number_input("Cluster", value=1)
        transactions = st.number_input("Transactions (scaled)", value=0.0)
    
    with col2:
        dcoilwtico = st.number_input("Oil Price (scaled)", value=0.0)
        year = st.number_input("Year", value=2017)
        month = st.number_input("Month", value=1, min_value=1, max_value=12)
        day = st.number_input("Day", value=1, min_value=1, max_value=31)

    # Dummy example: manually entering one-hot columns (real use: dynamic or backend)
    family_AUTOMOTIVE = st.checkbox("Is Family: AUTOMOTIVE?", value=True)

    submitted = st.form_submit_button("Predict Sales")

if submitted:
    # Create a base input dictionary with all features set to 0
    input_dict = {col: 0 for col in feature_names}

    # Update input_dict with user inputs
    input_dict.update({
        'store_nbr': store_nbr,
        'onpromotion': onpromotion,
        'cluster': cluster,
        'transactions': transactions,
        'dcoilwtico': dcoilwtico,
        'year': year,
        'month': month,
        'day': day,
        'family_AUTOMOTIVE': int(family_AUTOMOTIVE)
    })

    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])[feature_names]

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"📦 Predicted Sales: {prediction:.2f} units")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
