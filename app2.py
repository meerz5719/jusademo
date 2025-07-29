import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Load the stacked model
model = joblib.load("best_stacked_model_optuna1.pkl")

# Title
st.title("📦 Product Demand Forecasting App")
st.markdown("Predict future product demand based on store-level inputs.")

# Sidebar inputs
st.sidebar.header("🧾 Enter Input Features")

store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 55)))  # adjust based on your dataset
date_input = st.sidebar.date_input("Select Date", value=datetime.today())
promo = st.sidebar.selectbox("Is Promotion Active?", [0, 1])
transactions = st.sidebar.number_input("Transactions (optional)", min_value=0, value=1000, step=50)

# Build input dataframe
input_df = pd.DataFrame({
    "store_nbr": [store_nbr],
    "date": [date_input],
    "promo": [promo],
    "transactions": [transactions],
})

# ✅ Convert 'date' column to datetime format
input_df["date"] = pd.to_datetime(input_df["date"])

# Feature engineering
input_df["year"] = input_df["date"].dt.year
input_df["month"] = input_df["date"].dt.month
input_df["day"] = input_df["date"].dt.day
input_df["dayofweek"] = input_df["date"].dt.dayofweek
input_df.drop(columns=["date"], inplace=True)

# Add a button for prediction
if st.button("🔮 Predict Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Demand: **{round(prediction)} units**")

    # (Optional) Generate dummy next 10 days forecast
    st.subheader("📊 Demand Forecast for Next 10 Days (Synthetic Example)")
    future_dates = pd.date_range(date_input, periods=10)
    future_preds = [model.predict(input_df)[0] + np.random.randint(-10, 10) for _ in range(10)]
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Demand": future_preds})

    # Plot
    st.line_chart(forecast_df.set_index("Date"))

    # Download
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
