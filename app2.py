import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your trained XGBoost model
with open("XGB_model.pkl", "rb") as file:
    model = pickle.load(file)

# Dictionary for categorical encoding (if your model expects numeric values)
store_type_map = {"A": 0, "B": 1, "C": 2, "D": 3}
state_map = {'Bolivar': 0, 'Guayas': 1, 'Pichincha': 2}  # example, expand as per your data
product_family_map = {'AUTOMOTIVE': 0, 'BEVERAGES': 1, 'CLEANING': 2}  # example

st.title("🛒 Product Demand Forecasting (Sales Prediction)")

# User input
store_number = st.slider("Store Number", 0, 54, 1)
store_type = st.selectbox("Store Type", list(store_type_map.keys()))
cluster = st.selectbox("Cluster", list(range(1, 18)))

month = st.slider("Month", 1, 12, 1)
day = st.slider("Day", 1, 31, 1)
day_of_week = st.slider("Day of Week (0=Sun, 6=Sat)", 0, 6, 0)
year = st.number_input("Year (e.g., 2016)", min_value=2013, max_value=2025, value=2016)

product_family = st.selectbox("Product Family", list(product_family_map.keys()))
on_promo = st.number_input("Number of Items on Promotion", 0, 100, 0)
transactions = st.number_input("Number of Transactions", 0, 1000, 0)
crude_oil_price = st.number_input("Crude Oil Price", 0.0, 200.0, 50.0)
state = st.selectbox("State Where The Store Is Located", list(state_map.keys()))

# Prepare input for prediction
input_data = np.array([[
    store_number,
    store_type_map[store_type],
    cluster,
    product_family_map[product_family],
    on_promo,
    state_map[state],
    transactions,
    crude_oil_price,
    year,
    month,
    day,
    day_of_week
]])

# Predict
if st.button("🔮 Predict Demand"):
    prediction = model.predict(input_data)[0]
    prediction = np.maximum(prediction, 0)  # Ensure no negative sales
    st.success(f"📦 Predicted Sales for {product_family}: *{prediction:.2f} units*")
