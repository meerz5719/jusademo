import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Load the XGBoost model
model = joblib.load("XGB_model.pkl")

# Title
st.title("📦 Product Demand Forecasting App")
st.markdown("Predict future product demand using an XGBoost model trained on store-level features.")

# Sidebar inputs
st.sidebar.header("🧾 Input Features")

store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 55)))  # adjust to your actual range
date_input = st.sidebar.date_input("Select Date", value=datetime.today())
promo = st.sidebar.selectbox("Is Promotion Active?", [0, 1])
transactions = st.sidebar.number_input("Transactions", min_value=0, value=1000, step=50)

# Prepare input
input_df = pd.DataFrame({
    "store_nbr": [store_nbr],
    "date": [date_input],
    "promo": [promo],
    "transactions": [transactions],
})

# ✅ Convert to datetime and extract features
input_df["date"] = pd.to_datetime(input_df["date"])
input_df["year"] = input_df["date"].dt.year
input_df["month"] = input_df["date"].dt.month
input_df["day"] = input_df["date"].dt.day
input_df["dayofweek"] = input_df["date"].dt.dayofweek
input_df.drop(columns=["date"], inplace=True)

# ✅ Ensure feature order matches what model expects
expected_features = ['store_nbr', 'promo', 'transactions', 'year', 'month', 'day', 'dayofweek']

# Safety check
missing = [f for f in expected_features if f not in input_df.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
else:
    input_df = input_df[expected_features]

    # Button for prediction
    if st.button("🔮 Predict Demand"):
        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Demand: **{round(prediction)} units**")

        # Forecast next 10 days
        st.subheader("📊 Forecast for Next 10 Days")
        future_dates = pd.date_range(start=date_input, periods=10)
        future_forecasts = []

        for dt in future_dates:
            row = pd.DataFrame({
                "store_nbr": [store_nbr],
                "promo": [promo],
                "transactions": [transactions],
                "year": [dt.year],
                "month": [dt.month],
                "day": [dt.day],
                "dayofweek": [dt.weekday()],
            })
            row = row[expected_features]  # ensure correct order
            pred = model.predict(row)[0]
            future_forecasts.append(pred)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Demand": future_forecasts
        })

        # Plot
        st.line_chart(forecast_df.set_index("Date"))

        # Download
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
