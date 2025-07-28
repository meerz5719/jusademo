import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained stacked model
with open("best_STACKED_MODEL.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("📊 Product Demand Forecasting App")
st.markdown("Predict product **sales** based on store details, promotions, oil price, holidays, and more.")

# Sidebar input
st.sidebar.header("Input Features")

# Example: Input fields (expand based on your dataset)
store_nbr = st.sidebar.number_input("Store Number", min_value=1, max_value=100, value=1)
onpromotion = st.sidebar.number_input("Items on Promotion", min_value=0)
cluster = st.sidebar.number_input("Cluster", min_value=0, max_value=50, value=13)
transactions = st.sidebar.number_input("Transactions", min_value=0)
dcoilwtico = st.sidebar.number_input("Oil Price", min_value=0.0)
year = st.sidebar.selectbox("Year", [2013, 2014, 2015, 2016])
month = st.sidebar.selectbox("Month", list(range(1, 13)))
day = st.sidebar.selectbox("Day", list(range(1, 32)))

# Placeholder for one-hot encoded categorical inputs
# In production, ensure these match your training pipeline!
family_AUTOMOTIVE = st.sidebar.selectbox("Product Family: AUTOMOTIVE", [0.0, 1.0])

# You can expand with more dropdowns or checkboxes as needed...

# Create input DataFrame
input_data = pd.DataFrame({
    'store_nbr': [store_nbr],
    'onpromotion': [onpromotion],
    'cluster': [cluster],
    'transactions': [transactions],
    'dcoilwtico': [dcoilwtico],
    'year': [year],
    'month': [month],
    'day': [day],
    'family_AUTOMOTIVE': [family_AUTOMOTIVE],
    # Add remaining one-hot encoded or numerical columns
    # Must match your training data format exactly!
})

# Prediction
if st.button("Predict Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"📦 Predicted Sales: {prediction:.2f} units")

    # Optional: Visual feedback
    st.metric("Expected Sales", f"{prediction:.2f} units", help="Forecasted demand based on your input")

